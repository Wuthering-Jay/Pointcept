import os
import numpy as np
import laspy
from pathlib import Path
from typing import Union, List, Tuple, Optional, Literal
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


class LASTileProcessor:
    """
    专门针对LAS/LAZ点云文件的分块处理器。
    包含标签重映射、权重计算、标签过滤等功能，但移除了NPY格式保存功能。
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
                 require_labels: Optional[List[int]] = None):
        """
        初始化LAS分块处理器。
        
        Args:
            input_path: LAS文件路径或包含LAS文件的目录
            output_dir: 输出目录 (默认: 与输入文件同目录)
            window_size: (x_size, y_size) 矩形窗口大小 (LAS文件单位)
            min_points: 有效分块的最小点数阈值 (None表示跳过)
            max_points: 分块进一步细分的最大点数阈值 (None表示跳过)
            label_remap: 是否将标签重映射为连续值
            label_count: 是否统计标签数量并计算权重
            save_sample_weight: 是否保存样本权重
            require_labels: 需要保留的标签列表，其他标签会被设为-1
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
        
        # 标签处理相关属性
        if self.save_sample_weight and not self.label_count:
            print("Warning: `save_sample_weight` is True, so `label_count` is being automatically enabled to calculate class weights.")
            self.label_count = True
            
        from collections import defaultdict
        self.label_counts = defaultdict(int)
        self.file_label_counts = {}
        self.weights = {}
        self.sample_weights = {}
        self.label_map = {}
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        self.las_files = self._find_las_files()
    
    def _find_las_files(self) -> List[Path]:
        """查找输入路径中的所有LAS文件。"""
        if self.input_path.is_file() and self.input_path.suffix.lower() in ['.las', '.laz']:
            return [self.input_path]
        elif self.input_path.is_dir():
            return list(self.input_path.glob('*.las')) + list(self.input_path.glob('*.laz'))
        else:
            raise ValueError(f"输入路径 {self.input_path} 不是有效的LAS文件或目录")
    
    def process_all_files(self):
        """处理所有发现的LAS文件。"""
        # 标签收集和权重计算
        if self.label_remap or self.label_count:
            self._collect_labels()
        
        if self.label_count:
            self._calculate_weights()
            
        # 处理文件
        for las_file in tqdm(self.las_files, desc="处理文件", unit="文件"):
            print(f"正在处理 {las_file}...")
            self.process_file(las_file)
            
        # 保存标签相关信息
        if self.label_remap:
            self._save_label_map()
            
        if self.label_count:
            self._save_label_stats()
            
        if self.save_sample_weight:
            self._save_sample_weights_json()
    
    def process_file(self, las_file: Union[str, Path]):
        """处理单个LAS文件。"""
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
        将点云分割为矩形窗口并应用阈值。
        
        Args:
            points: Nx3的点坐标数组 (x, y, z)
            
        Returns:
            索引数组列表，每个表示一个分块中的点
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
        """
        对超过max_points阈值的分块进行细分。
        
        Args:
            points: Nx3的点坐标数组
            segments: 表示分块的索引数组列表
            
        Returns:
            处理后满足最大阈值的分块
        """
        # 快速识别需要细分的分块
        large_segment_indices = [i for i, segment in enumerate(segments) if len(segment) > self.max_points]
        
        if not large_segment_indices:
            return segments
        
        print(f"细分 {len(large_segment_indices)} 个超过 {self.max_points} 点的分块...")
        
        # 保留小分块，处理大分块
        result_segments = [segment for i, segment in enumerate(segments) if i not in large_segment_indices]
        large_segments = [segments[i] for i in large_segment_indices]
        
        # 递归细分分块的函数
        def process_segment(segment):
            if len(segment) <= self.max_points:
                return [segment]
            
            # 使用现有的细分逻辑
            segment_points = points[segment]
            ranges = np.ptp(segment_points[:, :2], axis=0)
            split_dim = np.argmax(ranges[:2])
            sorted_indices = np.argsort(segment_points[:, split_dim])
            
            # 分成两半
            mid = len(sorted_indices) // 2
            left_half = segment[sorted_indices[:mid]]
            right_half = segment[sorted_indices[mid:]]
            
            # 递归处理两半
            result = []
            result.extend(process_segment(left_half))
            result.extend(process_segment(right_half))
            return result
        
        # 使用多线程处理大分块
        max_workers = max(1, min(multiprocessing.cpu_count(), len(large_segments)))
        with tqdm(total=len(large_segments), desc="细分大分块", unit="分块") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_segment, segment) for segment in large_segments]
                for future in as_completed(futures):
                    result_segments.extend(future.result())
                    pbar.update(1)
        
        print(f"细分完成。处理后共 {len(result_segments)} 个分块。")
        return result_segments
    
    def apply_min_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        合并小于min_points阈值的分块到最近的分块。
        
        Args:
            points: Nx3的点坐标数组
            segments: 表示分块的索引数组列表
            
        Returns:
            处理后满足最小阈值的分块
        """
        if len(segments) <= 1:
            return segments
        
        print(f"合并少于 {self.min_points} 点的分块...")
        
        # 计算所有分块的质心
        centroids = np.array([np.mean(points[segment][:, :2], axis=0) for segment in segments])
        
        # 识别小分块
        small_segments = [i for i, segment in enumerate(segments) if len(segment) < self.min_points]
        
        if not small_segments:
            return segments
        
        # 按大小升序处理小分块（先合并最小的）
        small_segments.sort(key=lambda i: len(segments[i]))
        
        with tqdm(total=len(small_segments), desc="合并小分块", unit="分块") as pbar:
            for small_idx in small_segments:
                if small_idx >= len(segments) or len(segments[small_idx]) == 0:
                    pbar.update(1)
                    continue
                
                # 找到最近的非小分块
                nearest_idx = -1
                nearest_dist = float('inf')
                
                for i, segment in enumerate(segments):
                    if i != small_idx and len(segment) >= self.min_points:
                        dist = np.linalg.norm(centroids[i] - centroids[small_idx])
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_idx = i
                
                # 合并到最近的分块
                if nearest_idx != -1 and nearest_idx < len(segments):
                    segments[nearest_idx] = np.concatenate([segments[nearest_idx], segments[small_idx]])
                    segments[small_idx] = np.array([], dtype=int)
                
                pbar.update(1)
        
        # 移除空分块
        return [segment for segment in segments if len(segment) > 0]
    
    def save_segments_as_las(self, las_file: Path, las_data: laspy.LasData, segments: List[np.ndarray]):
        """
        将分块的点云保存为独立的LAS文件。
        
        Args:
            las_file: 原始LAS文件路径
            las_data: 原始LAS数据
            segments: 分块的索引数组列表
        """
        base_name = las_file.stem
        
        print(f"保存 {len(segments)} 个分块为LAS文件...")
        for i, segment_indices in tqdm(enumerate(segments), total=len(segments), desc="保存LAS分块", unit="文件"):
            # 创建header副本
            header = laspy.LasHeader(point_format=las_data.header.point_format, 
                                     version=las_data.header.version)
            
            # 复制坐标系信息
            header.x_scale = las_data.header.x_scale
            header.y_scale = las_data.header.y_scale
            header.z_scale = las_data.header.z_scale
            header.x_offset = las_data.header.x_offset
            header.y_offset = las_data.header.y_offset
            header.z_offset = las_data.header.z_offset
            
            # 创建新的LAS数据
            new_las = laspy.LasData(header)
            new_las.points = laspy.ScaleAwarePointRecord.zeros(len(segment_indices), header=header)
            
            # 复制分块中的点数据
            for dimension in las_data.point_format.dimension_names:
                setattr(new_las, dimension, getattr(las_data, dimension)[segment_indices])
            
            # 标签处理
            if hasattr(new_las, 'classification'):
                # 应用require_labels逻辑：将非require_labels的点设为-1
                if self.require_labels:
                    require_mask = np.zeros(len(new_las.classification), dtype=bool)
                    for label in self.require_labels:
                        require_mask |= (new_las.classification == label)
                    new_las.classification[~require_mask] = -1
                
                # 应用标签重映射
                if self.label_remap:
                    original_labels = new_las.classification.copy()
                    for original, remapped in self.label_map.items():
                        mask = (original_labels == original)
                        new_las.classification[mask] = remapped
            
            # 计算并保存样本权重
            if self.save_sample_weight and hasattr(new_las, 'classification'):
                segment_name = f"{base_name}_segment_{i:04d}"
                unique_labels_in_segment, counts = np.unique(new_las.classification, return_counts=True)
                
                segment_weights = {}
                for label, count in zip(unique_labels_in_segment, counts):
                    if int(label) in self.weights:
                        segment_weights[str(label)] = {
                            'weight': self.weights[int(label)],
                            'count': int(count)
                        }
                
                self.sample_weights[segment_name] = segment_weights
            
            # 复制VLR和CRS信息
            if hasattr(las_data.header, 'vlrs'):
                for vlr in las_data.header.vlrs:
                    new_las.header.vlrs.append(vlr)
                    
            if hasattr(las_data, 'crs'):
                new_las.crs = las_data.crs
                
            # 保存文件
            output_path = self.output_dir / f"{base_name}_segment_{i:04d}.las"
            new_las.write(output_path)
    
    def _collect_labels(self):
        """收集所有文件的标签信息，用于重映射和计数。"""
        print("收集所有文件的标签信息...")
        unique_labels = set()

        for las_file in tqdm(self.las_files, desc="扫描文件", unit="文件"):
            with laspy.open(las_file) as fh:
                las_data = fh.read()
                
                if hasattr(las_data, 'classification'):
                    # 应用require_labels过滤逻辑
                    if self.require_labels:
                        # 只统计require_labels中的标签
                        mask = np.zeros(len(las_data.classification), dtype=bool)
                        for label in self.require_labels:
                            mask |= (las_data.classification == label)
                        valid_labels = np.unique(las_data.classification[mask])
                        # 添加-1标签（非require_labels的点会被设为-1）
                        if not mask.all():  # 如果有点不在require_labels中
                            unique_labels.add(-1)
                    else:
                        # 如果没有指定require_labels，使用所有标签
                        mask = np.ones(len(las_data.classification), dtype=bool)
                        valid_labels = np.unique(las_data.classification[mask])
                    
                    unique_labels.update(valid_labels)
                    
                    if self.label_count:
                        # 计算实际的标签分布（包括-1）
                        modified_labels = las_data.classification.copy()
                        if self.require_labels:
                            # 将非require_labels的点设为-1
                            require_mask = np.zeros(len(modified_labels), dtype=bool)
                            for label in self.require_labels:
                                require_mask |= (modified_labels == label)
                            modified_labels[~require_mask] = -1
                        
                        unique_file_labels, counts = np.unique(modified_labels, return_counts=True)
                        file_counts = {int(label): int(count) for label, count in zip(unique_file_labels, counts)}
                        self.file_label_counts[las_file.name] = file_counts
                        
                        for label, count in file_counts.items():
                            self.label_counts[int(label)] += count
                            
        self.label_counts = dict(sorted(self.label_counts.items()))
        if self.label_remap:
            sorted_labels = sorted(unique_labels)
            self.label_map = {int(original): i for i, original in enumerate(sorted_labels)}
            print(f"创建标签映射: {self.label_map}")

    def _calculate_weights(self):
        """根据逆频率计算类别权重。"""
        if not self.label_counts:
            return
            
        total_points = sum(self.label_counts.values())
        self.weights = {}
        
        for label, count in self.label_counts.items():
            frequency = count / total_points
            self.weights[int(label)] = (1.0 / max(frequency**(1/2), 1e-6))

        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for label in self.weights:
                self.weights[label] /= weight_sum
            
        print(f"计算类别权重: {self.weights}")

    def _save_label_map(self):
        """保存标签映射到JSON文件。"""
        import json
        map_file = self.output_dir / "label_mapping.json"
        json_map = {str(k): int(v) for k, v in self.label_map.items()}
        
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(json_map, f, indent=2, ensure_ascii=False)
            
        print(f"标签映射已保存到 {map_file}")

    def _save_label_stats(self):
        """保存标签统计信息到JSON文件。"""
        import json
        stats_file = self.output_dir / "label_statistics.json"
        
        stats = {
            "total_counts": {str(k): int(v) for k, v in self.label_counts.items()},
            "file_counts": {
                file_name: {str(k): int(v) for k, v in counts.items()} 
                for file_name, counts in self.file_label_counts.items()
            },
            "weights": {str(k): float(v) for k, v in self.weights.items()}
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        print(f"标签统计信息已保存到 {stats_file}")

    def _save_sample_weights_json(self):
        """保存样本权重到JSON文件。"""
        import json
        weights_file = self.output_dir / "sample_weights.json"
        
        with open(weights_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_weights, f, indent=2, ensure_ascii=False)
            
        print(f"样本权重已保存到 {weights_file}")
    

def process_las_tiles(input_path: Union[str, Path],
                      output_dir: Union[str, Path] = None,
                      window_size: Tuple[float, float] = (50.0, 50.0),
                      min_points: Optional[int] = 1000,
                      max_points: Optional[int] = 5000,
                      label_remap: bool = False,
                      label_count: bool = False,
                      save_sample_weight: bool = False,
                      require_labels: Optional[List[int]] = None):
    """
    便捷函数：处理LAS/LAZ文件分块。
    
    Args:
        input_path: LAS文件路径或包含LAS文件的目录
        output_dir: 输出目录 (默认: 与输入文件同目录)
        window_size: (x_size, y_size) 矩形窗口大小
        min_points: 有效分块的最小点数阈值
        max_points: 分块进一步细分的最大点数阈值
        label_remap: 是否将标签重映射为连续值
        label_count: 是否统计标签数量并计算权重
        save_sample_weight: 是否保存样本权重
        require_labels: 需要保留的标签列表，其他标签会被设为-1
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
        require_labels=require_labels
    )
    processor.process_all_files()


if __name__ == "__main__":
    # 示例使用
    input_path = r"path/to/your/las/files"
    output_dir = r"path/to/output/directory"
    
    process_las_tiles(
        input_path=input_path,
        output_dir=output_dir,
        window_size=(20.0, 20.0),
        min_points=1000,
        max_points=8000,
        label_remap=True,  # 启用标签重映射为连续值
        label_count=True,  # 启用标签计数
        save_sample_weight=True,  # 计算样本权重
        require_labels=[1, 2, 3, 4, 5, 6]  # 只保留这些标签，其他设为-1
    )