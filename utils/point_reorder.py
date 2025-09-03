import os
import laspy
import numpy as np
from tqdm import tqdm
import gc  # For garbage collection

# --- Configuration ---
SOURCE_FOLDER = r"F:\WHU-Railways3D\rural_railway\tiles\output"
TARGET_FOLDER = r"F:\WHU-Railways3D\rural_railway\test"
OUTPUT_FOLDER = r"F:\WHU-Railways3D\rural_railway\result"
PRECISION_FACTOR = 1000  # Match to millimeter precision
CHUNK_SIZE = 5_000_000  # Process large files in chunks of this many points

def reorder_points_direct_matching(source_path, target_path, output_path):
    """
    Memory-efficient point matching and reordering with chunking for large files.
    """
    try:
        print(f"\n--- Processing file: {os.path.basename(source_path)} ---")
        
        # 1. Read source file and check if target file is too large
        print("Reading source file...")
        source_las = laspy.read(source_path)
        num_points = len(source_las.points)
        
        # Check target file size before loading
        target_info = laspy.open(target_path)
        if target_info.header.point_count > CHUNK_SIZE:
            print(f"Target file is large ({target_info.header.point_count:,} points). Using chunked processing.")
            return process_large_file(source_path, target_path, output_path)
        
        # For smaller files, use the direct approach
        print("Reading target file...")
        target_las = laspy.read(target_path)

        # Check point counts
        if len(source_las.points) != len(target_las.points):
            print(f"  [ERROR] Point count mismatch. Src: {len(source_las.points)}, Tgt: {len(target_las.points)}")
            return False
        
        # 2. Check if headers have matching scale/offset for fast path
        same_scale_offset = (
            source_las.header.x_scale == target_las.header.x_scale and
            source_las.header.y_scale == target_las.header.y_scale and
            source_las.header.z_scale == target_las.header.z_scale and
            source_las.header.x_offset == target_las.header.x_offset and
            source_las.header.y_offset == target_las.header.y_offset and
            source_las.header.z_offset == target_las.header.z_offset
        )
        
        if same_scale_offset:
            print("✓ Source and target share same scale and offset - using fast path")
            # Direct integer coordinate comparison
            source_points_raw = np.array([source_las.X, source_las.Y, source_las.Z]).T
            target_points_raw = np.array([target_las.X, target_las.Y, target_las.Z]).T
            
            # Build lookup table with integer coordinates
            target_dict = {}
            for i in tqdm(range(len(target_points_raw)), desc="Building target map"):
                key = tuple(target_points_raw[i])
                if key not in target_dict:
                    target_dict[key] = []
                target_dict[key].append(i)
            
            # Match source points to target points
            source_to_target_map = np.full(num_points, -1, dtype=np.int64)
            matched_count = 0
            
            for i in tqdm(range(len(source_points_raw)), desc="Matching points"):
                key = tuple(source_points_raw[i])
                if key in target_dict and target_dict[key]:
                    target_index = target_dict[key].pop(0)
                    source_to_target_map[i] = target_index
                    matched_count += 1
        
        else:
            print("Different scale/offset values detected - using standard matching")
            # Round coordinates to integers for stable dictionary keys
            target_keys = np.round(target_las.xyz * PRECISION_FACTOR).astype(np.int64)
            
            # Build lookup table
            target_dict = {}
            for i in tqdm(range(len(target_keys)), desc="Building target map"):
                key = tuple(target_keys[i])
                if key not in target_dict:
                    target_dict[key] = []
                target_dict[key].append(i)
            
            # Match source points to target points
            source_keys = np.round(source_las.xyz * PRECISION_FACTOR).astype(np.int64)
            source_to_target_map = np.full(num_points, -1, dtype=np.int64)
            matched_count = 0
            
            for i in tqdm(range(len(source_keys)), desc="Matching points"):
                key = tuple(source_keys[i])
                if key in target_dict and target_dict[key]:
                    target_index = target_dict[key].pop(0)
                    source_to_target_map[i] = target_index
                    matched_count += 1
        
        # Report match statistics
        match_percentage = (matched_count / num_points) * 100
        print(f"Matched {matched_count:,} points ({match_percentage:.2f}%)")
        
        if matched_count == 0:
            print("  [ERROR] No points were matched!")
            return False
            
        # 3. Generate final reordering indices
        new_order_indices = np.full(num_points, -1, dtype=np.int64)
        for src_idx, tgt_idx in enumerate(source_to_target_map):
            if tgt_idx != -1:
                new_order_indices[tgt_idx] = src_idx
        
        # Filter out unmatched indices
        valid_indices = new_order_indices[new_order_indices != -1]
        
        # 4. Save reordered point cloud
        print("Creating and writing new LAS file...")
        new_las = laspy.LasData(header=target_las.header)
        new_las.points = source_las.points[valid_indices]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_las.write(output_path)
        print(f"Processing complete! File saved to: {output_path}")
        
        # Clean up
        del source_las, target_las, target_dict, source_to_target_map, new_order_indices
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error processing file {os.path.basename(source_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_large_file(source_path, target_path, output_path):
    """
    Process large files with compatibility for older laspy versions.
    """
    try:
        print("Using chunked processing for large file...")
        
        # Read source file
        source_las = laspy.read(source_path)
        
        # Get target file header information (without loading all points)
        with laspy.open(target_path) as target_fh:
            target_header = target_fh.header
            target_point_count = target_header.point_count
            
            # Check point counts
            if len(source_las.points) != target_point_count:
                print(f"  [ERROR] Point count mismatch. Src: {len(source_las.points)}, Tgt: {target_point_count}")
                return False
                
            # Check if headers have matching scale/offset
            same_scale_offset = (
                source_las.header.x_scale == target_header.x_scale and
                source_las.header.y_scale == target_header.y_scale and
                source_las.header.z_scale == target_header.z_scale and
                source_las.header.x_offset == target_header.x_offset and
                source_las.header.y_offset == target_header.y_offset and
                source_las.header.z_offset == target_header.z_offset
            )
            
            # For older laspy versions, we need to read the full file
            print(f"Reading full target file ({target_point_count:,} points)...")
            target_las = target_fh.read()
            
            # Process in memory-efficient manner
            print("Building spatial lookup table...")
            target_dict = {}
            
            # Process in batches to avoid memory issues
            batch_size = min(CHUNK_SIZE, target_point_count)
            num_batches = (target_point_count + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, target_point_count)
                print(f"Processing batch {batch_idx+1}/{num_batches} (points {start_idx:,} to {end_idx:,})...")
                
                if same_scale_offset:
                    # Direct integer coordinate comparison
                    batch_points_raw = np.array([
                        target_las.X[start_idx:end_idx], 
                        target_las.Y[start_idx:end_idx], 
                        target_las.Z[start_idx:end_idx]
                    ]).T
                    
                    for i, point in enumerate(tqdm(batch_points_raw, desc="Building target map")):
                        key = tuple(point)
                        if key not in target_dict:
                            target_dict[key] = []
                        target_dict[key].append(start_idx + i)
                else:
                    # Use rounded coordinates
                    batch_xyz = target_las.xyz[start_idx:end_idx]
                    batch_keys = np.round(batch_xyz * PRECISION_FACTOR).astype(np.int64)
                    
                    for i, point in enumerate(tqdm(batch_keys, desc="Building target map")):
                        key = tuple(point)
                        if key not in target_dict:
                            target_dict[key] = []
                        target_dict[key].append(start_idx + i)
            
            # Match source points to target
            if same_scale_offset:
                print("✓ Using fast path matching (same scale/offset)")
                source_points_raw = np.array([source_las.X, source_las.Y, source_las.Z]).T
                source_keys = source_points_raw
            else:
                print("Using standard matching (different scale/offset)")
                source_keys = np.round(source_las.xyz * PRECISION_FACTOR).astype(np.int64)
            
            # Match points
            source_to_target_map = np.full(len(source_las.points), -1, dtype=np.int64)
            matched_count = 0
            
            for i in tqdm(range(len(source_keys)), desc="Matching points"):
                key = tuple(source_keys[i])
                if key in target_dict and target_dict[key]:
                    target_index = target_dict[key].pop(0)
                    source_to_target_map[i] = target_index
                    matched_count += 1
                    
            # Report match statistics
            match_percentage = (matched_count / len(source_las.points)) * 100
            print(f"Matched {matched_count:,} points ({match_percentage:.2f}%)")
            
            if matched_count == 0:
                print("  [ERROR] No points were matched!")
                return False
                
            # Generate reordering indices
            new_order_indices = np.full(len(source_las.points), -1, dtype=np.int64)
            for src_idx, tgt_idx in enumerate(source_to_target_map):
                if tgt_idx != -1:
                    new_order_indices[tgt_idx] = src_idx
            
            # Filter out unmatched indices
            valid_indices = new_order_indices[new_order_indices != -1]
            
            # Create new LAS file
            print("Creating and writing new LAS file...")
            new_las = laspy.LasData(header=target_header)
            new_las.points = source_las.points[valid_indices]
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_las.write(output_path)
            print(f"Processing complete! File saved to: {output_path}")
            
            # Clean up
            del target_las, source_las, target_dict, new_order_indices
            gc.collect()
            
            return True
            
    except Exception as e:
        print(f"Error processing large file {os.path.basename(source_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if not os.path.exists(SOURCE_FOLDER) or not os.path.exists(TARGET_FOLDER):
        print("Error: Source or target folder does not exist.")
        return
        
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    source_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.las', '.laz'))]
    
    if not source_files:
        print(f"No .las or .laz files found in source folder '{SOURCE_FOLDER}'.")
        return
        
    print(f"Found {len(source_files)} source files to process.")
    
    # Process files sequentially to avoid memory issues
    success_count = 0
    for filename in source_files:
        source_file_path = os.path.join(SOURCE_FOLDER, filename)
        target_file_path = os.path.join(TARGET_FOLDER, filename)
        output_file_path = os.path.join(OUTPUT_FOLDER, filename)
        
        if os.path.exists(target_file_path):
            print(f"\nProcessing {filename}...")
            success = reorder_points_direct_matching(source_file_path, target_file_path, output_file_path)
            if success:
                success_count += 1
            
            # Force garbage collection to free memory
            gc.collect()
        else:
            print(f"\n[WARNING] Corresponding file '{filename}' not found in target folder, skipped.")
    
    print(f"\nProcessing complete: {success_count}/{len(source_files)} files processed successfully.")

if __name__ == "__main__":
    main()