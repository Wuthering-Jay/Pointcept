import json
import numpy as np
import laspy
from pathlib import Path
from typing import Union, List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm


class LASMerger:
    def __init__(self, 
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 label_remap_file: Optional[Union[str, Path]] = None):
        """
        Initialize LAS point cloud merger (LAS format only).
        
        Args:
            input_path: Path to directory containing segmented LAS files
            output_dir: Directory to save merged files (default: same as input)
            label_remap_file: Path to label remapping JSON file (to convert back to original labels)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.label_remap_file = Path(label_remap_file) if label_remap_file else None
        
        # For label remapping
        self.inverse_label_map = {}  # Will store remapped_label -> original_label
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        # Load label remapping if provided
        if self.label_remap_file:
            self._load_label_map()
    
    def _load_label_map(self):
        """Load label mapping from file and create inverse mapping."""
        if not self.label_remap_file.exists():
            raise FileNotFoundError(f"Label remap file not found: {self.label_remap_file}")
        
        with open(self.label_remap_file) as f:
            label_map_data = json.load(f)
        
        # 支持新格式（包含 foreground_mapping 等元数据）和旧格式（直接是 {原始: 新} 映射）
        if "foreground_mapping" in label_map_data:
            # 新格式
            label_map = {int(k): int(v) for k, v in label_map_data["foreground_mapping"].items()}
        else:
            # 旧格式：直接是标签映射
            label_map = {int(k): int(v) for k, v in label_map_data.items()}
        
        # Create inverse mapping (新标签 -> 原始标签)
        self.inverse_label_map = {v: k for k, v in label_map.items()}
        print(f"Loaded inverse label mapping: {self.inverse_label_map}")
    
    def _find_segment_groups(self) -> Dict[str, List[Path]]:
        """
        Find all segment LAS files and group them by original file name.
        
        Returns:
            Dictionary mapping original file names to lists of segment files
        """
        segment_groups = defaultdict(list)
        
        # Find all LAS files
        las_files = list(self.input_path.glob("**/*.las")) + list(self.input_path.glob("**/*.laz"))
        
        # Group segments by original file name
        for file_path in las_files:
            # Assume naming format like "original_segment_XXX.las"
            parts = file_path.stem.split("_segment_")
            if len(parts) == 2:
                original_name = parts[0]
                segment_groups[original_name].append(file_path)
        
        # Sort segments by their index to ensure consistent order
        for original_name in segment_groups:
            segment_groups[original_name].sort(key=lambda p: int(p.name.split("_segment_")[1].split(".")[0]))
        
        return segment_groups
    
    def merge_all(self):
        """Merge all segmented point clouds back into original LAS files."""
        segment_groups = self._find_segment_groups()
        
        print(f"Found {len(segment_groups)} groups of segments to merge")
        for original_name, segments in tqdm(segment_groups.items(), desc="Merging files", unit="file"):
            print(f"Merging {len(segments)} segments for {original_name}")
            self.merge_las_segments(original_name, segments)
    
    def merge_las_segments(self, original_name: str, segment_files: List[Path]):
        """
        Merge LAS segment files back into a single LAS file.
        
        Args:
            original_name: Name of the original file
            segment_files: List of segment LAS files to merge
        """
        if not segment_files:
            print(f"  Warning: No segments found for {original_name}")
            return
        
        # Read first segment to get header and point format
        with laspy.open(segment_files[0]) as fh:
            first_segment = fh.read()
            header = laspy.LasHeader(point_format=first_segment.header.point_format,
                                     version=first_segment.header.version)
            
            # Explicitly copy scale and offset from first segment to ensure consistency
            header.x_scale = first_segment.header.x_scale
            header.y_scale = first_segment.header.y_scale
            header.z_scale = first_segment.header.z_scale
            header.x_offset = first_segment.header.x_offset
            header.y_offset = first_segment.header.y_offset
            header.z_offset = first_segment.header.z_offset
            
            # Copy VLRs and CRS
            if hasattr(first_segment.header, 'vlrs'):
                for vlr in first_segment.header.vlrs:
                    header.vlrs.append(vlr)
                    
            if hasattr(first_segment, 'crs'):
                header.crs = first_segment.crs
    
        # Count total points to allocate memory
        total_points = 0
        for segment_file in segment_files:
            with laspy.open(segment_file) as fh:
                total_points += len(fh.read())
        
        print(f"  Total points: {total_points}")
        
        # Create merged LAS file
        output_path = self.output_dir / f"{original_name}.las"
        merged_las = laspy.LasData(header)
        merged_las.points = laspy.ScaleAwarePointRecord.zeros(total_points, header=header)
        
        # Prepare inverse label map as NumPy array for vectorized operations
        remap_array = None
        if self.inverse_label_map:
            max_label = max(self.inverse_label_map.keys()) + 1
            remap_array = np.ones(max_label, dtype=np.int32) * -1
            for remapped_label, orig_label in self.inverse_label_map.items():
                remap_array[remapped_label] = orig_label
        
        # Process segments in batches to conserve memory
        BATCH_SIZE = 25  # Process this many segments before writing
        
        # Merge segments - process in small batches to avoid memory issues
        point_offset = 0
        for batch_start in range(0, len(segment_files), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(segment_files))
            batch_files = segment_files[batch_start:batch_end]
            
            print(f"  Processing batch {batch_start//BATCH_SIZE + 1}/{(len(segment_files)-1)//BATCH_SIZE + 1}")
            
            for segment_file in tqdm(batch_files, desc="  Processing segments", unit="segment"):
                with laspy.open(segment_file) as fh:
                    segment = fh.read()
                    segment_points = len(segment)
                    
                    # Copy all dimensions
                    for dimension in segment.point_format.dimension_names:
                        # Apply inverse label remapping if needed
                        if dimension == 'classification' and self.inverse_label_map and remap_array is not None:
                            original_labels = np.array(getattr(segment, dimension), dtype=np.int32)
                            remapped = np.copy(original_labels)
                            
                            # Only remap labels that are within bounds and have valid mapping
                            valid_mask = original_labels < len(remap_array)
                            valid_indices = original_labels[valid_mask]
                            mapped_values = remap_array[valid_indices]
                            has_mapping = mapped_values != -1
                            
                            # Apply remapping only where valid mapping exists
                            final_mask = np.zeros(len(original_labels), dtype=bool)
                            final_mask[valid_mask] = has_mapping
                            remapped[final_mask] = mapped_values[has_mapping]
                            
                            merged_las.classification[point_offset:point_offset+segment_points] = remapped.astype(np.uint8)
                        else:
                            dim_data = getattr(segment, dimension)
                            merged_data = getattr(merged_las, dimension)
                            merged_data[point_offset:point_offset+segment_points] = dim_data
                    
                    point_offset += segment_points
        
        # Update header statistics before saving
        # This ensures the header accurately reflects the data
        merged_las.update_header()
        
        # Save merged file
        merged_las.write(output_path)
        print(f"  Saved merged file to {output_path}")


def merge_las_segments(input_path: Union[str, Path], 
                      output_dir: Optional[Union[str, Path]] = None,
                      label_remap_file: Optional[Union[str, Path]] = None):
    """
    Merge segmented LAS point clouds back into original LAS files.
    
    Args:
        input_path: Path to directory containing segmented LAS files
        output_dir: Directory to save merged files (default: same as input)
        label_remap_file: Path to label remapping JSON file (to convert back to original labels)
    """
    merger = LASMerger(
        input_path=input_path,
        output_dir=output_dir,
        label_remap_file=label_remap_file
    )
    
    merger.merge_all()
    
    
if __name__ == "__main__":
    
    input_path = r"F:\WHU-Railways3D\urban_railway\tiles\test1"
    output_dir = r"F:\WHU-Railways3D\urban_railway\tiles\output"
    label_remap_file = r"F:\WHU-Railways3D\urban_railway\tiles\train\label_mapping.json"
    
    merge_las_segments(
        input_path=input_path,
        output_dir=output_dir,
        label_remap_file=label_remap_file
    )
