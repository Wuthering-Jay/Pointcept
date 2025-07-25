import os
import json
import numpy as np
import laspy
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any, Literal
from sklearn.neighbors import KDTree
from tqdm import tqdm
from collections import defaultdict, Counter

class LASProcessor:
    def __init__(self,
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 window_size: Tuple[float, float] = (50.0, 50.0),
                 min_points: Optional[int] = 1000,
                 max_points: Optional[int] = 5000,
                 ignore_labels: Optional[List[int]] = None,
                 require_labels: Optional[List[int]] = None,
                 label_remap: bool = False,
                 label_count: bool = False,
                 save_sample_weight: bool = False,
                 output_format: Literal["las", "npy"] = "las",
                 save_echo_ratio: bool = False,
                 save_color: bool = False,
                 save_intensity: bool = False,
                 save_normal: bool = False,
                 save_index: bool = False,
                 test_mode: bool = False):
        """
        Initialize LAS point cloud processor.
        
        Args:
            input_path: Path to LAS file or directory containing LAS files
            output_dir: Directory to save processed files (default: same as input)
            window_size: (x_size, y_size) for rectangular windows (in units of the LAS file)
            min_points: Minimum points threshold for a valid segment (None to skip)
            max_points: Maximum points threshold before further segmentation (None to skip)
            ignore_labels: List of label values to ignore during processing.
            require_labels: List of label values to keep during processing.
            label_remap: Whether to remap labels to continuous values
            label_count: Whether to count labels and calculate weights
            output_format: Output format ("las" or "npy")
            save_color: Whether to save color data in NPY format
            save_intensity: Whether to save intensity data in NPY format
            save_normal: Whether to save normal data in NPY format
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.window_size = window_size
        self.min_points = min_points
        self.max_points = max_points

        # 确保 ignore_labels 和 require_labels 不会同时使用
        if ignore_labels and require_labels:
            raise ValueError("`ignore_labels` and `require_labels` cannot be used at the same time.")

        self.ignore_labels = set(ignore_labels) if ignore_labels else set()
        self.require_labels = set(require_labels) if require_labels else set()
        
        self.label_remap = label_remap
        self.label_count = label_count
        self.save_sample_weight = save_sample_weight
        self.output_format = output_format
        self.save_echo_ratio = save_echo_ratio
        self.save_color = save_color
        self.save_intensity = save_intensity
        self.save_normal = save_normal
        self.save_index = save_index
        self.test_mode = test_mode
        
        # 如果需要保存样本权重，则必须开启标签计数以计算类别权重
        if self.save_sample_weight and not self.label_count:
            print("Warning: `save_sample_weight` is True, so `label_count` is being automatically enabled to calculate class weights.")
            self.label_count = True
        
        # For label remapping and counting
        self.label_map = {}
        self.label_counts = defaultdict(int)
        self.file_label_counts = {}
        self.weights = {}
        self.sample_weights = {}
        
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
        if self.label_remap or self.label_count:
            self._collect_labels()
        
        # 只有在 label_count 为 True 时才会计算权重
        if self.label_count:
            self._calculate_weights()

        for las_file in tqdm(self.las_files, desc="Processing files", unit="file",position=0):
            print(f"Processing {las_file}...")
            self.process_file(las_file)
            
        if self.label_remap:
            self._save_label_map()
            
        if self.label_count:
            self._save_label_stats()
            
        if self.save_sample_weight:
            self._save_sample_weights_json()
    
    def _collect_labels(self):
        """Collect all unique labels from all files for remapping and counting."""
        print("Collecting labels from all files...")
        unique_labels = set()

        for las_file in tqdm(self.las_files, desc="Scanning files", unit="file",position=0):
            with laspy.open(las_file) as fh:
                las_data = fh.read()
                
                if hasattr(las_data, 'classification'):
                    # 根据 ignore_labels 或 require_labels 创建掩码
                    mask = np.ones(len(las_data.classification), dtype=bool)
                    if self.ignore_labels:
                        for label in self.ignore_labels:
                            mask &= (las_data.classification != label)
                    elif self.require_labels:
                        # 如果指定了 require_labels，则只保留这些类别的点
                        required_mask = np.zeros_like(mask)
                        for label in self.require_labels:
                            required_mask |= (las_data.classification == label)
                        mask = required_mask
                    
                    valid_labels = np.unique(las_data.classification[mask])
                    unique_labels.update(valid_labels)
                    
                    if self.label_count:
                        unique_file_labels, counts = np.unique(las_data.classification[mask], return_counts=True)
                        file_counts = {int(label): int(count) for label, count in zip(unique_file_labels, counts)}
                        self.file_label_counts[las_file.name] = file_counts
                        
                        for label, count in file_counts.items():
                            self.label_counts[int(label)] += count
                            
        self.label_counts = dict(sorted(self.label_counts.items()))
        if self.label_remap:
            sorted_labels = sorted(unique_labels)
            self.label_map = {int(original): i for i, original in enumerate(sorted_labels)}
            print(f"Created label mapping: {self.label_map}")
    
    def _calculate_weights(self):
        """Calculate class weights based on inverse frequency."""
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
            
        print(f"Calculated class weights: {self.weights}")
    
    def _save_label_map(self):
        """Save label mapping to JSON file."""
        map_file = self.output_dir / "label_mapping.json"
        json_map = {str(k): int(v) for k, v in self.label_map.items()}
        
        with open(map_file, 'w') as f:
            json.dump(json_map, f, indent=2)
            
        print(f"Saved label mapping to {map_file}")
    
    def _save_label_stats(self):
        """Save label statistics to JSON file."""
        stats_file = self.output_dir / "label_statistics.json"
        
        stats = {
            "total_counts": {str(k): int(v) for k, v in self.label_counts.items()},
            "file_counts": {
                file_name: {str(k): int(v) for k, v in counts.items()} 
                for file_name, counts in self.file_label_counts.items()
            },
            "weights": {str(k): float(v) for k, v in self.weights.items()}
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"Saved label statistics to {stats_file}")
        
    def _save_sample_weights_json(self):
        """将收集到的所有样本权重保存到一个JSON文件中。"""
        if not self.sample_weights:
            print("Warning: No sample weights were calculated, JSON file will not be created.")
            return

        output_path = self.output_dir / "sample_weights.json"
        print(f"Saving sample weights for {len(self.sample_weights)} segments to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(self.sample_weights, f, indent=2)
            
        print("Successfully saved sample weights JSON file.")
    
    def process_file(self, las_file: Union[str, Path]):
        """Process a single LAS file."""
        las_file = Path(las_file)
        
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        
        # 创建掩码以选择要包含的点
        included_mask = np.ones(len(las_data.points), dtype=bool)
        if hasattr(las_data, 'classification'):
            if self.ignore_labels:
                for label in self.ignore_labels:
                    included_mask &= (las_data.classification != label)
            elif self.require_labels:
                # 如果指定了 require_labels，则只保留这些类别的点
                required_mask = np.zeros_like(included_mask)
                for label in self.require_labels:
                    required_mask |= (las_data.classification == label)
                included_mask = required_mask
        
        included_indices = np.where(included_mask)[0]
        if len(included_indices) == 0:
            print(f"  Warning: No valid points found in {las_file} after applying label filters.")
            return
        
        point_data = np.vstack((
            las_data.x[included_mask], 
            las_data.y[included_mask], 
            las_data.z[included_mask]
        )).transpose()
        
        segments = self.segment_point_cloud(point_data)
        original_segments = [included_indices[segment] for segment in segments]
        
        if self.output_format == "las":
            self.save_segments_as_las(las_file, las_data, original_segments)
        else: # "npy"
            self.save_segments_as_npy(las_file, las_data, original_segments)
            
    def segment_point_cloud(self, points: np.ndarray) -> List[np.ndarray]:
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
        
        # 后续阈值处理保持不变
        if self.max_points is not None:
            segments = self.apply_max_threshold(points, segments)
        if self.min_points is not None:
            segments = self.apply_min_threshold(points, segments)
            
        return segments
    
    # def segment_point_cloud(self, points: np.ndarray) -> List[np.ndarray]:
    #     """
    #     Segment the point cloud into rectangular windows and apply thresholds.
        
    #     Args:
    #         points: Nx3 array of point coordinates (x, y, z)
            
    #     Returns:
    #         List of index arrays, each representing points in a segment
    #     """
    #     # Get bounding box
    #     min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    #     max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        
    #     # Calculate number of windows (no overlap)
    #     x_size, y_size = self.window_size
        
    #     num_windows_x = max(1, int(np.ceil((max_x - min_x) / x_size)))
    #     num_windows_y = max(1, int(np.ceil((max_y - min_y) / y_size)))
        
    #     # Create initial segments by rectangular windows
    #     segments = []
    #     total_windows = num_windows_x * num_windows_y
        
    #     with tqdm(total=total_windows, desc="Creating segments", unit="segment",position=0) as pbar:
    #         for i in range(num_windows_x):
    #             for j in range(num_windows_y):
    #                 x_min = min_x + i * x_size
    #                 y_min = min_y + j * y_size
    #                 x_max = min(max_x, x_min + x_size)
    #                 y_max = min(max_y, y_min + y_size)
                    
    #                 # Find points in this window (exclusive on max boundaries to avoid duplicates)
    #                 # Except for the last window in each dimension
    #                 x_condition = (points[:, 0] >= x_min) & \
    #                             ((points[:, 0] < x_max) if i < num_windows_x - 1 else (points[:, 0] <= x_max))
    #                 y_condition = (points[:, 1] >= y_min) & \
    #                             ((points[:, 1] < y_max) if j < num_windows_y - 1 else (points[:, 1] <= y_max))
                    
    #                 mask = x_condition & y_condition
    #                 indices = np.where(mask)[0]
                    
    #                 if len(indices) > 0:
    #                     segments.append(indices)
                    
    #                 pbar.update(1)
        
    #     # Apply thresholds independently based on which ones are provided
    #     # If max_points is specified, subdivide large segments
    #     if self.max_points is not None:
    #         segments = self.apply_max_threshold(points, segments)
        
    #     # If min_points is specified, merge small segments
    #     if self.min_points is not None:
    #         segments = self.apply_min_threshold(points, segments)
            
    #     return segments
    
    def apply_max_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply max_points threshold to segments, subdividing large segments.
        
        Args:
            points: Nx3 array of point coordinates
            segments: List of index arrays representing segments
            
        Returns:
            Processed segments meeting the max threshold
        """
        # Handle segments above max_points threshold
        print(f"Subdividing segments with more than {self.max_points} points...")
        i = 0
        with tqdm(total=len(segments), desc="Checking max points", unit="segment",position=0) as pbar:
            while i < len(segments):
                if len(segments[i]) > self.max_points:
                    # Further divide this segment
                    new_segments = self.subdivide_segment(points, segments[i])
                    # Replace the current segment with the first new segment
                    segments[i] = new_segments[0]
                    # Add remaining new segments to the end
                    segments.extend(new_segments[1:])
                else:
                    i += 1
                pbar.update(1)
                # Update total to account for new segments
                pbar.total = len(segments)
                pbar.refresh()
                
        return segments
    
    def apply_min_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply min_points threshold to segments, merging small segments.
        
        Args:
            points: Nx3 array of point coordinates
            segments: List of index arrays representing segments
            
        Returns:
            Processed segments meeting the min threshold
        """
        # Handle segments below min_points threshold
        print(f"Merging segments with fewer than {self.min_points} points...")
        i = 0
        with tqdm(total=len(segments), desc="Checking min points", unit="segment",position=0) as pbar:
            while i < len(segments):
                if len(segments[i]) < self.min_points:
                    # Find nearest segment to merge with
                    nearest_segment_idx = self.find_nearest_segment(points, segments, i)
                    
                    if nearest_segment_idx != -1:
                        # Merge segments
                        segments[nearest_segment_idx] = np.concatenate([segments[nearest_segment_idx], segments[i]])
                        # Remove the merged segment
                        segments.pop(i)
                    else:
                        # No segment to merge with, keep as is
                        i += 1
                else:
                    i += 1
                    
                pbar.update(1)
                # Update total to account for merged segments
                pbar.total = len(segments)
                pbar.refresh()
                
        return segments
    
    def subdivide_segment(self, points: np.ndarray, segment: np.ndarray) -> List[np.ndarray]:
        """
        Subdivide a segment that exceeds max_points threshold.
        
        Args:
            points: Nx3 array of point coordinates
            segment: Indices of points in the segment to subdivide
            
        Returns:
            List of new segments after subdivision
        """
        # Get segment points
        segment_points = points[segment]
        
        # Find the dimension with largest spread (x or y only)
        ranges = np.ptp(segment_points[:, :2], axis=0)
        split_dim = np.argmax(ranges[:2])
        
        # Sort points along that dimension
        sorted_indices = np.argsort(segment_points[:, split_dim])
        
        # Split into two halves
        mid = len(sorted_indices) // 2
        left_half = segment[sorted_indices[:mid]]
        right_half = segment[sorted_indices[mid:]]
        
        result = []
        
        # Recursively split if still above threshold
        if len(left_half) > self.max_points:
            result.extend(self.subdivide_segment(points, left_half))
        else:
            result.append(left_half)
            
        if len(right_half) > self.max_points:
            result.extend(self.subdivide_segment(points, right_half))
        else:
            result.append(right_half)
            
        return result
    
    def find_nearest_segment(self, points: np.ndarray, segments: List[np.ndarray], segment_idx: int) -> int:
        """
        Find the nearest segment to merge with.
        
        Args:
            points: Nx3 array of point coordinates
            segments: List of index arrays representing segments
            segment_idx: Index of the segment to find neighbors for
            
        Returns:
            Index of the nearest segment, or -1 if no suitable segment found
        """
        if len(segments) <= 1:
            return -1
            
        # Calculate segment centroids
        source_centroid = np.mean(points[segments[segment_idx]][:, :2], axis=0)
        segment_centroids = []
        
        for i, segment in enumerate(segments):
            if i != segment_idx:
                segment_centroids.append((i, np.mean(points[segment][:, :2], axis=0)))
        
        # Find the nearest segment
        nearest_idx = -1
        nearest_dist = float('inf')
        
        for idx, centroid in segment_centroids:
            dist = np.linalg.norm(centroid - source_centroid)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
                
        return nearest_idx
    
    def save_segments_as_las(self, las_file: Path, las_data: laspy.LasData, segments: List[np.ndarray]):
        """
        Save segmented point clouds to separate LAS files.
        
        Args:
            las_file: Original LAS file path
            las_data: Original LAS data
            segments: List of index arrays for segments
        """
        base_name = las_file.stem
        
        print(f"Saving {len(segments)} segments as LAS files...")
        for i, segment_indices in tqdm(enumerate(segments), total=len(segments), desc="Saving LAS segments", unit="file",position=0):
            # Calculate sample weight for this segment if enabled
            if self.save_sample_weight:
                segment_name = f"{base_name}_segment_{i:04d}"
                current_sample_weight = 0.0
                if hasattr(las_data, 'classification'):
                    original_segment_labels = las_data.classification[segment_indices]
                    unique_labels_in_segment = np.unique(original_segment_labels)
                    current_sample_weight = sum(self.weights.get(label, 0.0) for label in unique_labels_in_segment)
            
            # Store the weight in the dictionary for later JSON saving
            self.sample_weights[segment_name] = float(current_sample_weight)
            
            # Create a copy of the header
            header = laspy.LasHeader(point_format=las_data.header.point_format, 
                                     version=las_data.header.version)
            
            # Create a new LAS data with the correct point count
            new_las = laspy.LasData(header)
            
            # Create points array with the correct size
            new_las.points = laspy.ScaleAwarePointRecord.zeros(
                len(segment_indices),
                header=header
            )
            
            # Copy points from this segment
            for dimension in las_data.point_format.dimension_names:
                # Apply label remapping to classification dimension if enabled
                if dimension == 'classification':
                    if self.test_mode:
                        # In test mode, use zeros for classification
                        setattr(new_las, dimension, np.zeros(len(segment_indices), dtype=np.int32))
                    else:
                        # Original classification handling logic
                        original_labels = getattr(las_data, dimension)[segment_indices]
                        if self.label_remap:
                            max_label_val = np.max(original_labels) + 1
                            remap_array = np.ones(max_label_val, dtype=np.int32) * -1
                            
                            for orig_label, new_label in self.label_map.items():
                                if orig_label < max_label_val:
                                    remap_array[orig_label] = new_label
                            
                            for label in self.ignore_labels:
                                if label < max_label_val:
                                    remap_array[label] = label
                            
                            remapped_labels = remap_array[original_labels]
                            setattr(new_las, dimension, remapped_labels)
                        else:
                            setattr(new_las, dimension, original_labels)
                else:
                    setattr(new_las, dimension, getattr(las_data, dimension)[segment_indices])
            
            # Also copy the header spatial reference system if it exists
            if hasattr(las_data.header, 'vlrs'):
                for vlr in las_data.header.vlrs:
                    new_las.header.vlrs.append(vlr)
                    
            # Copy CRS information if available
            if hasattr(las_data, 'crs'):
                new_las.crs = las_data.crs
                
            # Save to file
            output_path = self.output_dir / f"{base_name}_segment_{i:04d}.las"
            new_las.write(output_path)

    def save_segments_as_npy(self, las_file: Path, las_data: laspy.LasData, segments: List[np.ndarray]):
        """
        Save segmented point clouds to NPY files in separate folders.
        
        Args:
            las_file: Original LAS file path
            las_data: Original LAS data
            segments: List of index arrays for segments
        """
        base_name = las_file.stem
        
        print(f"Saving {len(segments)} segments as NPY files...")
        for i, segment_indices in tqdm(enumerate(segments), total=len(segments), desc="Saving NPY segments", unit="folder",position=0):
            # Create segment folder
            segment_name = f"{base_name}_segment_{i:04d}"
            segment_folder = self.output_dir / f"{base_name}_segment_{i:04d}"
            segment_folder.mkdir(exist_ok=True)
            
            if self.save_sample_weight:
                current_sample_weight = 0.0
                if hasattr(las_data, 'classification'):
                    original_segment_labels = las_data.classification[segment_indices]
                    unique_labels_in_segment = np.unique(original_segment_labels)
                    current_sample_weight = sum(self.weights.get(label, 0.0) for label in unique_labels_in_segment)
                
                # 将权重存入字典，而不是保存为文件
                self.sample_weights[segment_name] = float(current_sample_weight)

            # Always save coordinates (required)
            coords = np.vstack((
                las_data.x[segment_indices],
                las_data.y[segment_indices],
                las_data.z[segment_indices]
            )).T  # Transpose to get [N, 3]
            
            np.save(segment_folder / "coord.npy", coords)
            
            # Always save original indices (required)
            if self.save_index:
                np.save(segment_folder / "index.npy", segment_indices)
            
            # Handle labels (classification) - required, default to zeros if not present
            if self.test_mode:
                # In test mode, use zeros for all segments
                segment_labels = np.zeros(len(segment_indices), dtype=np.int32)
            else:
                if hasattr(las_data, 'classification'):
                    labels = las_data.classification[segment_indices]
                    
                    # Apply remapping if enabled
                    if self.label_remap:
                        # Vectorized remapping
                        max_label_val = np.max(labels) + 1
                        remap_array = np.zeros(max_label_val, dtype=np.int32)
                        
                        # Fill in the mapping values
                        for orig_label, new_label in self.label_map.items():
                            if orig_label < max_label_val:
                                remap_array[orig_label] = new_label
                        
                        # Apply remapping
                        segment_labels = remap_array[labels]
                    else:
                        segment_labels = labels
                else:
                    # No classification in input, use zeros
                    segment_labels = np.zeros(len(segment_indices), dtype=np.int32)
            
            np.save(segment_folder / "segment.npy", segment_labels)
            
            # Save Echo Ration(Return Number/Number of Returns) if requested and available
            if self.save_echo_ratio and all(hasattr(las_data, dim) for dim in ['return_number', 'number_of_returns']):
                echo_ratio=las_data.return_number[segment_indices]/(las_data.number_of_returns[segment_indices]+1e-6)
                np.save(segment_folder / "echo_ratio.npy", echo_ratio)
            elif self.save_echo_ratio:
                echo_ratio=np.ones(len(segment_indices),dtype=np.uint16)
                np.save(segment_folder / "echo_ratio.npy", echo_ratio)
            
            # Save color if requested and available
            if self.save_color and all(hasattr(las_data, dim) for dim in ['red', 'green', 'blue']):
                colors = np.vstack((
                    las_data.red[segment_indices],
                    las_data.green[segment_indices],
                    las_data.blue[segment_indices]
                )).T  # Transpose to get [N, 3]
                
                np.save(segment_folder / "color.npy", colors)
            elif self.save_color:
                # User requested color but it's not available - save zeros
                colors = np.zeros((len(segment_indices), 3), dtype=np.uint16)
                np.save(segment_folder / "color.npy", colors)
                
            # Save intensity if requested
            if self.save_intensity:
                if hasattr(las_data, 'intensity'):
                    intensity = las_data.intensity[segment_indices].astype(np.float32)/255.0
                else:
                    # User requested intensity but it's not available - save zeros
                    intensity = np.zeros(len(segment_indices), dtype=np.float32)
                
                np.save(segment_folder / "intensity.npy", intensity)
            
            # Save normals if requested
            if self.save_normal:
                if all(hasattr(las_data, dim) for dim in ['nx', 'ny', 'nz']):
                    normals = np.vstack((
                        las_data.nx[segment_indices],
                        las_data.ny[segment_indices],
                        las_data.nz[segment_indices]
                    )).T  # Transpose to get [N, 3]
                else:
                    # User requested normals but they're not available - save zeros
                    normals = np.zeros((len(segment_indices), 3), dtype=np.float32)
                
                np.save(segment_folder / "normal.npy", normals)

def process_las_files(input_path, output_dir=None, window_size=(50.0, 50.0), 
                      min_points=None, max_points=None, 
                      ignore_labels=None, require_labels=None, label_remap=False, label_count=False,
                      save_sample_weight=False,
                      output_format="las", 
                      save_echo_ratio=False, save_color=False, save_intensity=False, save_normal=False, save_index=False,
                      test_mode=False):
    processor = LASProcessor(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        ignore_labels=ignore_labels,
        require_labels=require_labels,
        label_remap=label_remap,
        label_count=label_count,
        save_sample_weight=save_sample_weight,
        output_format=output_format,
        save_echo_ratio=save_echo_ratio,
        save_color=save_color,
        save_intensity=save_intensity,
        save_normal=save_normal,
        save_index=save_index,
        test_mode=test_mode
    )
    processor.process_all_files()
    
    
if __name__ == "__main__":
    
    input_path=r"E:\data\railway"
    output_dir=r"E:\data\railway\tiles\train"
    window_size=(10., 10.)
    min_points=4096*2
    max_points=4096*16*4
    ignore_labels=[]
    require_labels=None
    # ignore_labels=None
    # require_labels=[2,5,6,9,11,13,15]
    label_remap=True
    label_count=True
    save_sample_weight=True 
    output_format="las"
    save_echo_ratio=False
    save_color=False
    save_intensity=True
    save_normal=False
    save_index=False
    test_mode=False
    
    process_las_files(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        ignore_labels=ignore_labels,
        require_labels=require_labels,
        label_remap=label_remap,
        label_count=label_count,
        save_sample_weight=save_sample_weight,
        output_format=output_format,
        save_echo_ratio=save_echo_ratio,
        save_color=save_color,
        save_intensity=save_intensity,
        save_normal=save_normal,
        save_index=save_index,
        test_mode=test_mode
    )