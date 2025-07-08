import laspy
import numpy as np
import time
from pathlib import Path


# -------------------------------------------------

def process_las_pair(original_path, updated_path, output_path):
    """
    处理单个LAS文件对的核心函数。
    此函数无需修改，因为它已能处理 .las 和 .laz 文件。
    """
    try:
        print(f"  - 步骤 1: 读取 '{original_path.name}' 的头文件以获取标准参数...")
        with laspy.open(original_path) as f_orig:
            if not f_orig.header.point_count > 0:
                print(f"  - 警告: 原始文件 '{original_path.name}' 为空，已跳过。")
                return
            std_scale = f_orig.header.scales
            std_offset = f_orig.header.offsets

        print(f"  - 步骤 2: 读取更新文件 '{updated_path.name}' 并构建查找表...")
        with laspy.open(updated_path) as f_upd:
            if not f_upd.header.point_count > 0:
                print(f"  - 警告: 更新文件 '{updated_path.name}' 为空，已跳过。")
                return
            updated_las = f_upd.read()
            int_x = np.round((updated_las.x - std_offset[0]) / std_scale[0]).astype(np.int64)
            int_y = np.round((updated_las.y - std_offset[1]) / std_scale[1]).astype(np.int64)
            int_z = np.round((updated_las.z - std_offset[2]) / std_scale[2]).astype(np.int64)
            classification_map = {
                coord: classification
                for coord, classification in zip(zip(int_x, int_y, int_z), updated_las.classification)
            }

        print(f"  - 步骤 3: 读取原始文件 '{original_path.name}' 并更新分类码...")
        with laspy.open(original_path, mode='r') as original_file:
            original_las = original_file.read()
            orig_int_x = np.round((original_las.x - std_offset[0]) / std_scale[0]).astype(np.int64)
            orig_int_y = np.round((original_las.y - std_offset[1]) / std_scale[1]).astype(np.int64)
            orig_int_z = np.round((original_las.z - std_offset[2]) / std_scale[2]).astype(np.int64)

            updated_count = 0
            new_classifications = original_las.classification.copy()
            for i in range(len(original_las.points)):
                coord_key = (orig_int_x[i], orig_int_y[i], orig_int_z[i])
                if coord_key in classification_map:
                    new_classifications[i] = classification_map[coord_key]
                    updated_count += 1
            
            original_las.classification = new_classifications
            print(f"  - 更新统计: 在 {len(original_las.points)} 个点中，成功匹配并更新 {updated_count} 个。")

        print(f"  - 步骤 4: 写入结果到 '{output_path.name}'...")
        # laspy会根据输出文件名的后缀(.laz)自动进行压缩
        with laspy.open(output_path, mode='w', header=original_las.header) as output_file:
            output_file.write_points(original_las.points)
        
        print(f"  - 成功: '{original_path.name}' 处理完成。")

    except Exception as e:
        print(f"  - 错误: 处理 '{original_path.name}' 时发生严重错误: {e}")
        print("  - 已跳过此文件对。")


def main():
    """
    批量处理的主函数，现在支持忽略后缀进行文件名匹配。
    """
    print("--- 开始批量处理 (智能匹配模式) ---")
    start_time = time.time()

    original_p = Path(ORIGINAL_FOLDER)
    updated_p = Path(UPDATED_FOLDER)
    output_p = Path(OUTPUT_FOLDER)

    if not original_p.is_dir() or not updated_p.is_dir():
        print(f"错误: 请确保原始文件夹 '{ORIGINAL_FOLDER}' 和更新文件夹 '{UPDATED_FOLDER}' 都存在。")
        return

    print(f"输出文件夹将被创建于: '{output_p}'")
    output_p.mkdir(parents=True, exist_ok=True)
    
    # 获取原始文件夹中所有 .las 和 .laz 文件
    files_to_process = [p for p in original_p.iterdir() if p.suffix.lower() in ['.las', '.laz']]
    
    if not files_to_process:
        print("在原始文件夹中没有找到任何 .las 或 .laz 文件。")
        return
        
    print(f"在原始文件夹中找到 {len(files_to_process)} 个LAS/LAZ文件，开始匹配和处理...")
    print("-" * 20)

    total_files = len(files_to_process)
    processed_count = 0
    skipped_count = 0

    for i, original_file_path in enumerate(files_to_process):
        # 【新逻辑】获取不带后缀的主文件名 (e.g., "tile_001")
        basename = original_file_path.stem
        print(f"[{i+1}/{total_files}] 正在检查主文件名: '{basename}' (来自 '{original_file_path.name}')")

        # 【新逻辑】在更新文件夹中查找匹配的主文件名，不限后缀
        updated_file_path = None
        potential_updated_las = updated_p / f"{basename}.las"
        potential_updated_laz = updated_p / f"{basename}.laz"

        if potential_updated_las.exists():
            updated_file_path = potential_updated_las
        elif potential_updated_laz.exists():
            updated_file_path = potential_updated_laz
        
        # 如果找到了匹配文件 (无论后缀是.las还是.laz)
        if updated_file_path:
            print(f"  - 找到匹配文件: '{updated_file_path.name}'")
            # 【新逻辑】输出文件名和原始文件名保持一致
            output_file_path = output_p / original_file_path.name
            
            process_las_pair(original_file_path, updated_file_path, output_file_path)
            processed_count += 1
        else:
            print(f"  - 跳过: 在更新文件夹中找不到主文件名为 '{basename}' 的 .las 或 .laz 文件。")
            skipped_count += 1
        print("-" * 20)

    end_time = time.time()
    print("--- 批量处理完成 ---")
    print(f"总计: {total_files} 个文件。")
    print(f"成功处理: {processed_count} 个文件对。")
    print(f"跳过: {skipped_count} 个文件（因缺少对应文件）。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == '__main__':
    
    # --- 【1. 配置区】请修改为您自己的文件夹路径 ---
    ORIGINAL_FOLDER = r'E:\data\WHU-Railway3D-las\urban_railway\test'
    UPDATED_FOLDER = r'E:\data\WHU-Railway3D-las\urban_railway\npy\pred'
    OUTPUT_FOLDER = r'E:\data\WHU-Railway3D-las\urban_railway\output'

    main()