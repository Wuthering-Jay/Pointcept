import os
import json
import numpy as np
import laspy
from pathlib import Path
from typing import Union, List, Dict, Optional, Literal, Tuple
from collections import defaultdict
from tqdm import tqdm

class LASMerger:
    def __init__(self, 
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 input_format: Literal["las", "npy"] = "las",
                 label_file: str = "segment",
                 label_remap_file: Optional[Union[str, Path]] = None,
                 template_las_file: Optional[Union[str, Path]] = None):
        """
        Initialize LAS point cloud merger.
        
        Args:
            input_path: Path to directory containing segmented LAS files or NPY folders
            output_dir: Directory to save merged files (default: same as input)
            input_format: Format of input segments ("las" or "npy")
            label_file: Name of the NPY file containing labels (without .npy extension)
            label_remap_file: Path to label remapping JSON file (to convert back to original labels)
            template_las_file: Path to a template LAS file to use for header information (for NPY mode)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.input_format = input_format
        self.label_file = label_file
        self.label_remap_file = Path(label_remap_file) if label_remap_file else None
        self.template_las_file = Path(template_las_file) if template_las_file else None
        
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
            label_map = json.load(f)
        
        # Convert keys from strings back to integers
        label_map = {int(k): int(v) for k, v in label_map.items()}
        
        # Create inverse mapping
        self.inverse_label_map = {v: k for k, v in label_map.items()}
        print(f"Loaded inverse label mapping: {self.inverse_label_map}")
    
    def _find_segment_groups(self) -> Dict[str, List[Path]]:
        """
        Find all segment files/folders and group them by original file name.
        
        Returns:
            Dictionary mapping original file names to lists of segment files/folders
        """
        segment_groups = defaultdict(list)
        
        if self.input_format == "las":
            # Find all LAS files
            las_files = list(self.input_path.glob("**/*.las")) + list(self.input_path.glob("**/*.laz"))
            
            # Group segments by original file name
            for file_path in las_files:
                # Assume naming format like "original_segment_XXX.las"
                parts = file_path.stem.split("_segment_")
                if len(parts) == 2:
                    original_name = parts[0]
                    segment_groups[original_name].append(file_path)
        else:  # NPY format
            # Find all segment folders
            segment_folders = [d for d in self.input_path.glob("*_segment_*") if d.is_dir()]
            
            # Group segments by original file name
            for folder_path in segment_folders:
                # Assume naming format like "original_segment_XXX"
                parts = folder_path.name.split("_segment_")
                if len(parts) == 2:
                    original_name = parts[0]
                    segment_groups[original_name].append(folder_path)
        
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
            if self.input_format == "las":
                self.merge_las_segments(original_name, segments)
            else:  # NPY format
                self.merge_npy_segments(original_name, segments)
    
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
        
        # Count total points to allocate memory
        total_points = 0
        for segment_file in segment_files:
            with laspy.open(segment_file) as fh:
                total_points += len(fh.read())
        
        print(f"  Total points: {total_points}")
        
        # Create merged LAS file
        merged_las = laspy.LasData(header)
        merged_las.points = laspy.ScaleAwarePointRecord.zeros(total_points, header=header)
        
        # Merge segments
        point_offset = 0
        for segment_file in tqdm(segment_files, desc="  Reading segments", unit="segment"):
            with laspy.open(segment_file) as fh:
                segment = fh.read()
                segment_points = len(segment)
                
                # Copy all dimensions
                for dimension in segment.point_format.dimension_names:
                    # Apply inverse label remapping if needed
                    if dimension == 'classification' and self.inverse_label_map:
                        original_labels = getattr(segment, dimension)
                        # Use vectorized approach for remapping
                        remapped = np.array([self.inverse_label_map.get(int(label), int(label)) 
                                           for label in original_labels])
                        merged_las.classification[point_offset:point_offset+segment_points] = remapped
                    else:
                        dim_data = getattr(segment, dimension)
                        merged_data = getattr(merged_las, dimension)
                        merged_data[point_offset:point_offset+segment_points] = dim_data
                
                point_offset += segment_points
        
        # Copy VLRs and CRS
        if hasattr(first_segment.header, 'vlrs'):
            for vlr in first_segment.header.vlrs:
                merged_las.header.vlrs.append(vlr)
                
        if hasattr(first_segment, 'crs'):
            merged_las.crs = first_segment.crs
        
        # Save merged file
        output_path = self.output_dir / f"{original_name}.las"
        merged_las.write(output_path)
        print(f"  Saved merged file to {output_path}")
    
    def _calculate_header_bounds(self, segment_folders: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate min/max bounds and appropriate scale for coordinates.
        
        Args:
            segment_folders: List of segment folders containing coord.npy files
        
        Returns:
            Tuple of (mins, maxs, scales) for x,y,z coordinates
        """
        # Initialize min/max arrays
        mins = np.array([float('inf'), float('inf'), float('inf')])
        maxs = np.array([float('-inf'), float('-inf'), float('-inf')])
        
        print("  Calculating coordinate bounds...")
        # Process each segment to find global min/max
        for folder in tqdm(segment_folders, desc="  Scanning coordinates", unit="segment"):
            coord_file = folder / "coord.npy"
            if coord_file.exists():
                coords = np.load(coord_file)
                folder_mins = np.min(coords, axis=0)
                folder_maxs = np.max(coords, axis=0)
                
                mins = np.minimum(mins, folder_mins)
                maxs = np.maximum(maxs, folder_maxs)
        
        # Calculate appropriate scales based on range and precision
        ranges = maxs - mins
        # Use scale that gives at least 0.1mm precision
        scales = np.maximum(ranges / (2**31 - 1), 0.0001)  # LAS uses signed 32-bit integers
        
        print(f"  Coordinate ranges: X: {ranges[0]:.2f}, Y: {ranges[1]:.2f}, Z: {ranges[2]:.2f}")
        print(f"  Calculated scales: X: {scales[0]:.6f}, Y: {scales[1]:.6f}, Z: {scales[2]:.6f}")
        
        return mins, maxs, scales
    
    def merge_npy_segments(self, original_name: str, segment_folders: List[Path]):
        """
        Merge NPY segment folders back into a single LAS file.
        
        Args:
            original_name: Name of the original file
            segment_folders: List of segment folders to merge
        """
        if not segment_folders:
            print(f"  Warning: No segments found for {original_name}")
            return
        
        # Get template LAS file for header information
        header = None
        if self.template_las_file and self.template_las_file.exists():
            with laspy.open(self.template_las_file) as fh:
                template = fh.read()
                header = laspy.LasHeader(point_format=template.header.point_format,
                                        version=template.header.version)
                
                # Copy template header's scale and offset
                header.x_scale = template.header.x_scale
                header.y_scale = template.header.y_scale
                header.z_scale = template.header.z_scale
                header.x_offset = template.header.x_offset
                header.y_offset = template.header.y_offset
                header.z_offset = template.header.z_offset
                
                print(f"  Using template file scales: X: {header.x_scale}, Y: {header.y_scale}, Z: {header.z_scale}")
                print(f"  Using template file offsets: X: {header.x_offset}, Y: {header.y_offset}, Z: {header.z_offset}")
        else:
            # Create a default header with point format 7 (includes RGB)
            header = laspy.LasHeader(point_format=7, version="1.4")
            
            # Calculate appropriate scale and offset based on data
            mins, maxs, scales = self._calculate_header_bounds(segment_folders)
            
            # Set scale and offset for better precision
            header.x_scale = scales[0]
            header.y_scale = scales[1]
            header.z_scale = scales[2]
            header.x_offset = mins[0]
            header.y_offset = mins[1]
            header.z_offset = mins[2]
            
            print(f"  Auto-set offsets: X: {header.x_offset}, Y: {header.y_offset}, Z: {header.z_offset}")
            
        # Count total points to allocate memory
        total_points = 0
        for segment_folder in segment_folders:
            coord_file = segment_folder / "coord.npy"
            if coord_file.exists():
                coords = np.load(coord_file)
                total_points += coords.shape[0]
            
        print(f"  Total points: {total_points}")
        
        if total_points == 0:
            print(f"  Warning: No valid points found in segments for {original_name}")
            return
        
        # Create merged LAS file
        merged_las = laspy.LasData(header)
        merged_las.points = laspy.ScaleAwarePointRecord.zeros(total_points, header=header)
        
        # Check what optional data is present in the first segment
        first_segment = segment_folders[0]
        has_color = (first_segment / "color.npy").exists()
        has_intensity = (first_segment / "intensity.npy").exists()
        has_normal = (first_segment / "normal.npy").exists()
        
        # Merge segments
        point_offset = 0
        for segment_folder in tqdm(segment_folders, desc="  Reading segments", unit="segment"):
            # Required files
            coord_file = segment_folder / "coord.npy"
            label_file_path = segment_folder / f"{self.label_file}.npy"
            
            if not coord_file.exists():
                print(f"  Warning: coord.npy not found in {segment_folder}, skipping")
                continue
                
            # Load coordinates
            coords = np.load(coord_file)
            segment_points = coords.shape[0]
            
            try:
                # Copy coordinates
                merged_las.x[point_offset:point_offset+segment_points] = coords[:, 0]
                merged_las.y[point_offset:point_offset+segment_points] = coords[:, 1]
                merged_las.z[point_offset:point_offset+segment_points] = coords[:, 2]
            except OverflowError as e:
                # Provide more detailed error message with diagnostics
                print(f"  Error: Coordinates out of range for LAS header scale/offset!")
                print(f"  Header x_scale: {header.x_scale}, y_scale: {header.y_scale}, z_scale: {header.z_scale}")
                print(f"  Header x_offset: {header.x_offset}, y_offset: {header.y_offset}, z_offset: {header.z_offset}")
                print(f"  Coordinate ranges - X: [{np.min(coords[:, 0])}, {np.max(coords[:, 0])}], " 
                      f"Y: [{np.min(coords[:, 1])}, {np.max(coords[:, 1])}], "
                      f"Z: [{np.min(coords[:, 2])}, {np.max(coords[:, 2])}]")
                print(f"  Try providing a template LAS file or increasing scale values.")
                raise
            
            # Copy labels
            if label_file_path.exists():
                labels = np.load(label_file_path)
                
                # Apply inverse label remapping if needed
                if self.inverse_label_map:
                    # Use vectorized approach for remapping
                    max_label_val = np.max(labels) + 1
                    remap_array = np.ones(max_label_val, dtype=np.int32) * -1
                    
                    # Fill in the mapping values
                    for remapped_label, orig_label in self.inverse_label_map.items():
                        if remapped_label < max_label_val:
                            remap_array[remapped_label] = orig_label
                    
                    # Apply remapping
                    remapped_labels = remap_array[labels]
                    # Handle any unmapped values
                    unmapped_mask = remapped_labels == -1
                    if np.any(unmapped_mask):
                        remapped_labels[unmapped_mask] = labels[unmapped_mask]
                        
                    merged_las.classification[point_offset:point_offset+segment_points] = remapped_labels
                else:
                    merged_las.classification[point_offset:point_offset+segment_points] = labels
            else:
                print(f"  Warning: {self.label_file}.npy not found in {segment_folder}, using zeros")
                merged_las.classification[point_offset:point_offset+segment_points] = 0
            
            # Copy color if available
            if has_color and (segment_folder / "color.npy").exists():
                colors = np.load(segment_folder / "color.npy")
                merged_las.red[point_offset:point_offset+segment_points] = colors[:, 0]
                merged_las.green[point_offset:point_offset+segment_points] = colors[:, 1]
                merged_las.blue[point_offset:point_offset+segment_points] = colors[:, 2]
            
            # Copy intensity if available
            if has_intensity and (segment_folder / "intensity.npy").exists():
                intensity = np.load(segment_folder / "intensity.npy")
                merged_las.intensity[point_offset:point_offset+segment_points] = intensity
            
            # Copy normals if available (typically stored as extra dimensions)
            if has_normal and (segment_folder / "normal.npy").exists() and hasattr(merged_las, 'nx'):
                normals = np.load(segment_folder / "normal.npy")
                merged_las.nx[point_offset:point_offset+segment_points] = normals[:, 0]
                merged_las.ny[point_offset:point_offset+segment_points] = normals[:, 1]
                merged_las.nz[point_offset:point_offset+segment_points] = normals[:, 2]
            
            point_offset += segment_points
        
        # Save merged file
        output_path = self.output_dir / f"{original_name}.las"
        merged_las.write(output_path)
        print(f"  Saved merged file to {output_path}")


def merge_las_segments(input_path, output_dir=None, input_format="las", 
                      label_file="segment", label_remap_file=None,
                      template_las_file=None):
    """
    Merge segmented point clouds back into original LAS files.
    
    Args:
        input_path: Path to directory containing segmented LAS files or NPY folders
        output_dir: Directory to save merged files (default: same as input)
        input_format: Format of input segments ("las" or "npy")
        label_file: Name of the NPY file containing labels (without .npy extension)
        label_remap_file: Path to label remapping JSON file (to convert back to original labels)
        template_las_file: Path to a template LAS file to use for header information (for NPY mode)
    """
    merger = LASMerger(
        input_path=input_path,
        output_dir=output_dir,
        input_format=input_format,
        label_file=label_file,
        label_remap_file=label_remap_file,
        template_las_file=template_las_file
    )
    
    merger.merge_all()
    
    
if __name__ == "__main__":
    
    input_path=r"E:\data\WHU-Railway3D-las\urban_railway\npy\test"
    output_dir=r"E:\data\WHU-Railway3D-las\urban_railway\npy\pred"
    input_format="npy"
    label_file="pred"
    label_remap_file=r"E:\data\WHU-Railway3D-las\urban_railway\npy\train\label_mapping.json"
    template_las_file=None
    
    merge_las_segments(
        input_path=input_path,
        output_dir=output_dir,
        input_format=input_format,
        label_file=label_file,
        label_remap_file=label_remap_file,
        template_las_file=template_las_file
    )