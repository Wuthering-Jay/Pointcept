import os
import json
import numpy as np
import laspy
from pathlib import Path
from typing import Union, List, Tuple, Optional, Literal
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


class LASTileProcessor:
    """
    Specialized tile processor for LAS/LAZ point cloud files.
    Includes label remapping, weight calculation, label filtering, and trash bin mechanism.
    """
    
    def __init__(self,
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 window_size: Tuple[float, float] = (50.0, 50.0),
                 min_points: Optional[int] = 1000,
                 max_points: Optional[int] = 5000,
                 label_remap: bool = False,
                 label_count: bool = False,
                 save_sample_weight: bool = False,
                 require_labels: Optional[List[int]] = None,
                 use_trash_bin: bool = False,
                 trash_bin_label: int = 0):
        """
        Initialize LAS tile processor.
        
        Args:
            input_path: LAS file path or directory containing LAS files
            output_dir: Output directory (default: same as input file directory)
            window_size: (x_size, y_size) rectangular window size (in LAS file units)
            min_points: Minimum point threshold for valid tiles (None to skip)
            max_points: Maximum point threshold for further subdivision (None to skip)
            label_remap: Whether to remap labels to continuous values
            label_count: Whether to count labels and calculate weights
            save_sample_weight: Whether to save sample weights
            require_labels: List of labels to keep (foreground classes)
            use_trash_bin: Whether to enable trash bin mechanism
            trash_bin_label: Label value for trash bin (default: 0, recommended)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.window_size = window_size
        self.min_points = min_points
        self.max_points = max_points
        self.label_remap = label_remap
        self.label_count = label_count
        self.save_sample_weight = save_sample_weight
        self.require_labels = require_labels
        self.use_trash_bin = use_trash_bin
        self.trash_bin_label = trash_bin_label
        
        # 验证垃圾桶机制配置
        if self.use_trash_bin:
            if not self.require_labels:
                raise ValueError("use_trash_bin=True requires require_labels to be specified!")
            if self.trash_bin_label != 0:
                print(f"Warning: trash_bin_label={self.trash_bin_label} (recommended value is 0)")
            print(f"Trash bin mechanism enabled: label {self.trash_bin_label}")
        
        # 标签处理相关属性
        if self.save_sample_weight and not self.label_count:
            print("Warning: `save_sample_weight` is True, so `label_count` is being automatically enabled.")
            self.label_count = True
            
        from collections import defaultdict
        self.label_counts = defaultdict(int)
        self.file_label_counts = {}
        self.weights = {}
        self.sample_weights = {}
        self.label_map = {}
        self.background_labels = set()  # 记录背景类标签
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        self.las_files = self._find_las_files()
    
    def _find_las_files(self) -> List[Path]:
        """Find all LAS files in the input path."""
        if self.input_path.is_file() and self.input_path.suffix.lower() in ['.las', '.laz']:
            return [self.input_path]
        elif self.input_path.is_dir():
            return list(self.input_path.glob('*.las')) + list(self.input_path.glob('*.laz'))
        else:
            raise ValueError(f"Input path {self.input_path} is not a valid LAS file or directory")
    
    def process_all_files(self):
        """Process all discovered LAS files."""
        # 标签收集和权重计算
        if self.label_remap or self.label_count:
            self._collect_labels()
        
        if self.label_count:
            self._calculate_weights()
            
        # 处理文件
        for las_file in tqdm(self.las_files, desc="Processing files", unit="file"):
            print(f"Processing {las_file}...")
            self.process_file(las_file)
            
        # 保存标签相关信息
        if self.label_remap:
            self._save_label_map()
            
        if self.label_count:
            self._save_label_stats()
            
        if self.save_sample_weight:
            self._save_sample_weights_json()
    
    def process_file(self, las_file: Union[str, Path]):
        """Process a single LAS file."""
        las_file = Path(las_file)
        
        # 读取LAS文件
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        
        # 提取点云坐标
        point_data = np.vstack((
            las_data.x, 
            las_data.y, 
            las_data.z
        )).transpose()
        
        # 分块处理
        segments = self.segment_point_cloud(point_data)
        
        # 保存分块结果为LAS文件
        self.save_segments_as_las(las_file, las_data, segments)
    
    def segment_point_cloud(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Segment point cloud into rectangular windows and apply thresholds.
        
        Args:
            points: Nx3 point coordinate array (x, y, z)
            
        Returns:
            List of index arrays, each representing points in a segment
        """
        # 获取边界框
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        x_size, y_size = self.window_size
        
        num_windows_x = max(1, int(np.ceil((max_x - min_x) / x_size)))
        num_windows_y = max(1, int(np.ceil((max_y - min_y) / y_size)))
        
        # 计算每个点所属的窗口索引
        x_bins = np.clip(((points[:, 0] - min_x) / x_size).astype(int), 0, num_windows_x - 1)
        y_bins = np.clip(((points[:, 1] - min_y) / y_size).astype(int), 0, num_windows_y - 1)
        
        # 组合窗口索引
        window_ids = x_bins * num_windows_y + y_bins
        
        # 分组
        unique_ids, indices = np.unique(window_ids, return_inverse=True)
        segments = [np.where(indices == i)[0] for i in range(len(unique_ids))]
        
        # 应用阈值处理
        if self.max_points is not None:
            segments = self.apply_max_threshold(points, segments)
        if self.min_points is not None:
            segments = self.apply_min_threshold(points, segments)
            
        return segments
    
    def apply_max_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """Subdivide segments that exceed max_points threshold."""
        large_segment_indices = [i for i, segment in enumerate(segments) if len(segment) > self.max_points]
        
        if not large_segment_indices:
            return segments
        
        print(f"Subdividing {len(large_segment_indices)} segments exceeding {self.max_points} points...")
        
        result_segments = [segment for i, segment in enumerate(segments) if i not in large_segment_indices]
        large_segments = [segments[i] for i in large_segment_indices]
        
        def process_segment(segment):
            if len(segment) <= self.max_points:
                return [segment]
            
            segment_points = points[segment]
            ranges = np.ptp(segment_points[:, :2], axis=0)
            split_dim = np.argmax(ranges[:2])
            sorted_indices = np.argsort(segment_points[:, split_dim])
            
            mid = len(sorted_indices) // 2
            left_half = segment[sorted_indices[:mid]]
            right_half = segment[sorted_indices[mid:]]
            
            result = []
            result.extend(process_segment(left_half))
            result.extend(process_segment(right_half))
            return result
        
        max_workers = max(1, min(multiprocessing.cpu_count(), len(large_segments)))
        with tqdm(total=len(large_segments), desc="Subdividing large segments", unit="segment") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_segment, segment) for segment in large_segments]
                for future in as_completed(futures):
                    result_segments.extend(future.result())
                    pbar.update(1)
        
        print(f"Subdivision complete. Total {len(result_segments)} segments after processing.")
        return result_segments
    
    def apply_min_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """Merge segments smaller than min_points threshold to nearest segments."""
        if len(segments) <= 1:
            return segments
        
        print(f"Merging segments with fewer than {self.min_points} points...")
        
        centroids = np.array([np.mean(points[segment][:, :2], axis=0) for segment in segments])
        small_segments = [i for i, segment in enumerate(segments) if len(segment) < self.min_points]
        
        if not small_segments:
            return segments
        
        small_segments.sort(key=lambda i: len(segments[i]))
        
        with tqdm(total=len(small_segments), desc="Merging small segments", unit="segment") as pbar:
            for small_idx in small_segments:
                if small_idx >= len(segments) or len(segments[small_idx]) == 0:
                    pbar.update(1)
                    continue
                
                nearest_idx = -1
                nearest_dist = float('inf')
                
                for i, segment in enumerate(segments):
                    if i != small_idx and len(segment) >= self.min_points:
                        dist = np.linalg.norm(centroids[i] - centroids[small_idx])
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_idx = i
                
                if nearest_idx != -1 and nearest_idx < len(segments):
                    segments[nearest_idx] = np.concatenate([segments[nearest_idx], segments[small_idx]])
                    segments[small_idx] = np.array([], dtype=int)
                
                pbar.update(1)
        
        return [segment for segment in segments if len(segment) > 0]
    
    def save_segments_as_las(self, las_file: Path, las_data: laspy.LasData, segments: List[np.ndarray]):
        """
        Save segmented point clouds as separate LAS files.
        
        Args:
            las_file: Original LAS file path
            las_data: Original LAS data
            segments: List of index arrays for segments
        """
        base_name = las_file.stem
        
        print(f"Saving {len(segments)} segments as LAS files...")
        for i, segment_indices in tqdm(enumerate(segments), total=len(segments), desc="Saving LAS segments", unit="file"):
            # 标签过滤 - 在创建LAS之前先过滤
            if hasattr(las_data, 'classification'):
                segment_labels = las_data.classification[segment_indices]
                
                if self.use_trash_bin:
                    # 垃圾桶模式：保留所有点
                    valid_mask = np.ones(len(segment_indices), dtype=bool)
                else:
                    # 传统模式：只保留前景类点
                    if self.require_labels:
                        valid_mask = np.zeros(len(segment_labels), dtype=bool)
                        for label in self.require_labels:
                            valid_mask |= (segment_labels == label)
                        
                        # 如果过滤后没有点了，跳过这个segment
                        if not np.any(valid_mask):
                            continue
                    else:
                        # 没有指定require_labels，保留所有点
                        valid_mask = np.ones(len(segment_indices), dtype=bool)
                
                # 应用过滤
                filtered_indices = segment_indices[valid_mask]
            else:
                # 没有classification字段，保留所有点
                filtered_indices = segment_indices
            
            # 创建LAS文件
            header = laspy.LasHeader(point_format=las_data.header.point_format, 
                                     version=las_data.header.version)
            
            header.x_scale = las_data.header.x_scale
            header.y_scale = las_data.header.y_scale
            header.z_scale = las_data.header.z_scale
            header.x_offset = las_data.header.x_offset
            header.y_offset = las_data.header.y_offset
            header.z_offset = las_data.header.z_offset
            
            new_las = laspy.LasData(header)
            new_las.points = laspy.ScaleAwarePointRecord.zeros(len(filtered_indices), header=header)
            
            # 复制所有维度数据
            for dimension in las_data.point_format.dimension_names:
                setattr(new_las, dimension, getattr(las_data, dimension)[filtered_indices])
            
            # 标签重映射
            if hasattr(new_las, 'classification') and self.label_remap:
                if self.use_trash_bin:
                    # 垃圾桶模式：背景类映射到trash_bin_label，前景类按映射表
                    original_labels = np.array(new_las.classification, dtype=np.int32)
                    remapped_labels = np.full(len(original_labels), self.trash_bin_label, dtype=np.int32)
                    
                    # 前景类应用映射
                    for original, remapped in self.label_map.items():
                        if original != self.trash_bin_label:
                            mask = (original_labels == original)
                            remapped_labels[mask] = remapped
                    
                    new_las.classification[:] = remapped_labels.astype(np.uint8)
                else:
                    # 传统模式：只对保留的前景类进行映射
                    original_labels = np.array(new_las.classification, dtype=np.int32)
                    remapped_labels = np.copy(original_labels)
                    
                    for original, remapped in self.label_map.items():
                        mask = (original_labels == original)
                        remapped_labels[mask] = remapped
                    
                    new_las.classification[:] = remapped_labels.astype(np.uint8)
            
            # 计算并保存样本权重
            if self.save_sample_weight:
                segment_name = f"{base_name}_segment_{i:04d}"
                current_sample_weight = 0.0
                
                if hasattr(new_las, 'classification'):
                    # 获取segment中的唯一标签（已经过remap处理）
                    unique_labels_in_segment = np.unique(new_las.classification)
                    # 对每个唯一标签的权重求和
                    current_sample_weight = sum(
                        self.weights.get(int(label), 0.0) 
                        for label in unique_labels_in_segment
                    )
                
                # 存储权重
                self.sample_weights[segment_name] = float(current_sample_weight)
            
            # 复制元数据
            if hasattr(las_data.header, 'vlrs'):
                for vlr in las_data.header.vlrs:
                    new_las.header.vlrs.append(vlr)
                    
            if hasattr(las_data, 'crs'):
                new_las.crs = las_data.crs
            
            # Update header statistics before saving
            # This ensures the header accurately reflects the data
            new_las.update_header()
                
            output_path = self.output_dir / f"{base_name}_segment_{i:04d}.las"
            new_las.write(output_path)

    def _collect_labels(self):
        """收集所有文件的标签信息，用于重映射和计数。"""
        print("Collecting labels from all files...")
        foreground_labels = set()

        for las_file in tqdm(self.las_files, desc="Scanning files", unit="file"):
            with laspy.open(las_file) as fh:
                las_data = fh.read()
                
                if not hasattr(las_data, 'classification'):
                    continue
                
                all_labels = las_data.classification
                
                if self.require_labels:
                    # 识别前景和背景
                    foreground_mask = np.zeros(len(all_labels), dtype=bool)
                    for label in self.require_labels:
                        foreground_mask |= (all_labels == label)
                    
                    # 记录背景类（用于调试）
                    background_labels_in_file = np.unique(all_labels[~foreground_mask])
                    self.background_labels.update(background_labels_in_file)
                    
                    # 只收集前景类用于remap
                    valid_labels = np.unique(all_labels[foreground_mask])
                    foreground_labels.update(valid_labels)
                else:
                    # 如果没有指定require_labels，所有标签都是前景
                    valid_labels = np.unique(all_labels)
                    foreground_labels.update(valid_labels)
                
                if self.label_count:
                    # 标签计数 - 修正：传统模式也只统计前景类
                    if self.require_labels:
                        # 无论哪种模式，都只统计前景类
                        foreground_mask = np.zeros(len(all_labels), dtype=bool)
                        for label in self.require_labels:
                            foreground_mask |= (all_labels == label)
                        
                        fg_labels = all_labels[foreground_mask]
                        unique_fg, counts_fg = np.unique(fg_labels, return_counts=True)
                        file_counts = {int(label): int(count) 
                                     for label, count in zip(unique_fg, counts_fg)}
                    else:
                        # 没有指定require_labels，统计所有标签
                        unique_file_labels, counts = np.unique(all_labels, return_counts=True)
                        file_counts = {int(label): int(count) 
                                     for label, count in zip(unique_file_labels, counts)}
                    
                    self.file_label_counts[las_file.name] = file_counts
                    
                    for label, count in file_counts.items():
                        self.label_counts[int(label)] += count
        
        self.label_counts = dict(sorted(self.label_counts.items()))
        
        if self.label_remap:
            sorted_labels = sorted(foreground_labels)
            if self.use_trash_bin:
                # 垃圾桶占据trash_bin_label，前景类从trash_bin_label+1开始映射
                # 或者如果trash_bin_label=0，前景类从1开始
                if self.trash_bin_label == 0:
                    self.label_map = {int(orig): i+1 for i, orig in enumerate(sorted_labels)}
                else:
                    # 如果trash_bin_label不是0，需要避开它
                    self.label_map = {}
                    next_id = 0
                    for orig in sorted_labels:
                        if next_id == self.trash_bin_label:
                            next_id += 1
                        self.label_map[int(orig)] = next_id
                        next_id += 1
                
                print(f"Foreground label mapping: {self.label_map}")
                print(f"Background labels (will map to {self.trash_bin_label}): {sorted(self.background_labels)}")
            else:
                # 传统模式，从0开始
                self.label_map = {int(orig): i for i, orig in enumerate(sorted_labels)}
                print(f"Label mapping: {self.label_map}")

    def _calculate_weights(self):
        """根据逆频率计算类别权重（只对前景类计算）。"""
        if not self.label_counts:
            return
        
        # 修正：两种模式都只对前景类计算权重
        total_points = sum(self.label_counts.values())
        self.weights = {}
        
        if self.use_trash_bin:
            # 垃圾桶模式：垃圾桶权重为0
            self.weights[self.trash_bin_label] = 0.0
        
        # 只对label_counts中的类别（已经是前景类）计算权重
        for label, count in self.label_counts.items():
            frequency = count / total_points
            
            if self.label_remap:
                # 使用remap后的标签
                remapped_label = self.label_map.get(label, label)
            else:
                remapped_label = label
            
            self.weights[remapped_label] = (1.0 / max(frequency**(1/2), 1e-6))
        
        # 归一化（排除垃圾桶）
        if self.use_trash_bin:
            weight_sum = sum(w for label, w in self.weights.items() 
                           if label != self.trash_bin_label)
        else:
            weight_sum = sum(self.weights.values())
        
        if weight_sum > 0:
            for label in self.weights:
                if not self.use_trash_bin or label != self.trash_bin_label:
                    self.weights[label] /= weight_sum
        
        print(f"Calculated class weights: {self.weights}")

    def _save_label_map(self):
        """保存标签映射到JSON文件。"""
        import json
        map_file = self.output_dir / "label_mapping.json"
        
        # 构建完整的映射信息
        mapping_data = {
            "use_trash_bin": self.use_trash_bin,
            "trash_bin_label": self.trash_bin_label,
            "foreground_mapping": {str(k): int(v) for k, v in self.label_map.items()},
        }
        
        if self.use_trash_bin:
            mapping_data["background_classes"] = sorted([int(x) for x in self.background_labels])
            mapping_data["num_foreground_classes"] = len(self.label_map)
            mapping_data["total_classes"] = len(self.label_map) + 1  # +1 for trash bin
        
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved label mapping to {map_file}")

    def _save_label_stats(self):
        """保存标签统计信息到JSON文件。"""
        import json
        stats_file = self.output_dir / "label_statistics.json"
        
        stats = {
            "use_trash_bin": self.use_trash_bin,
            "trash_bin_label": self.trash_bin_label if self.use_trash_bin else None,
            "total_counts": {str(k): int(v) for k, v in self.label_counts.items()},
            "file_counts": {
                file_name: {str(k): int(v) for k, v in counts.items()} 
                for file_name, counts in self.file_label_counts.items()
            },
            "weights": {str(k): float(v) for k, v in self.weights.items()}
        }
        
        if self.use_trash_bin:
            stats["background_classes"] = sorted([int(x) for x in self.background_labels])
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        print(f"Saved label statistics to {stats_file}")

    def _save_sample_weights_json(self):
        """保存样本权重到JSON文件。"""
        import json
        weights_file = self.output_dir / "sample_weights.json"
        
        with open(weights_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_weights, f, indent=2, ensure_ascii=False)
            
        print(f"Saved sample weights to {weights_file}")
    

def process_las_tiles(input_path: Union[str, Path],
                      output_dir: Union[str, Path] = None,
                      window_size: Tuple[float, float] = (50.0, 50.0),
                      min_points: Optional[int] = 1000,
                      max_points: Optional[int] = 5000,
                      label_remap: bool = False,
                      label_count: bool = False,
                      save_sample_weight: bool = False,
                      require_labels: Optional[List[int]] = None,
                      use_trash_bin: bool = False,
                      trash_bin_label: int = 0):
    """
    Convenience function: Process LAS/LAZ file tiling.
    
    Args:
        input_path: LAS file path or directory containing LAS files
        output_dir: Output directory (default: same as input file directory)
        window_size: (x_size, y_size) rectangular window size
        min_points: Minimum point threshold for valid tiles
        max_points: Maximum point threshold for further subdivision
        label_remap: Whether to remap labels to continuous values
        label_count: Whether to count labels and calculate weights
        save_sample_weight: Whether to save sample weights
        require_labels: List of labels to keep (foreground classes)
        use_trash_bin: Whether to enable trash bin mechanism
        trash_bin_label: Label value for trash bin (default: 0)
    """
    processor = LASTileProcessor(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        label_remap=label_remap,
        label_count=label_count,
        save_sample_weight=save_sample_weight,
        require_labels=require_labels,
        use_trash_bin=use_trash_bin,
        trash_bin_label=trash_bin_label
    )
    processor.process_all_files()


if __name__ == "__main__":
    # 示例1: 传统模式（背景类设为无效值）
    # input_path = r"E:\data\DALES\dales_las\test"
    # output_dir = r"E:\data\DALES\dales_las\test_traditional"
    
    # process_las_tiles(
    #     input_path=input_path,
    #     output_dir=output_dir,
    #     window_size=(20.0, 20.0),
    #     min_points=1000,
    #     max_points=8000,
    #     label_remap=True,
    #     label_count=True,
    #     save_sample_weight=True,
    #     require_labels=[1, 2, 3, 4, 5, 6],
    #     use_trash_bin=False  # 传统模式
    # )
    
    # 示例2: 垃圾桶模式（背景类标记为0）
    input_path = r"E:\data\DALES\dales_las\train"
    output_dir = r"E:\data\DALES\dales_las\tile\train"
    
    process_las_tiles(
        input_path=input_path,
        output_dir=output_dir,
        window_size=(50.0, 50.0),
        min_points=4096,
        max_points=None,
        label_remap=True,  # 启用标签重映射
        label_count=True,  # 启用标签计数
        save_sample_weight=True,  # 计算样本权重
        require_labels=[1,2,3,4,5,6,7,8],  # 前景类列表
    )