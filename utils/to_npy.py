import laspy
import numpy as np
from pathlib import Path
import time


# -------------------------------------------------

def convert_classification_to_npy(las_path, npy_path):
    """
    读取一个LAS/LAZ文件，提取其classification数据，
    并将其保存为一个NumPy格式的.npy文件（数据类型为uint8）。

    Args:
        las_path (Path): 输入的LAS/LAZ文件路径对象。
        npy_path (Path): 输出的.npy文件路径对象。

    Returns:
        bool: 如果成功则返回True，否则返回False。
    """
    try:
        print(f"  - 正在读取: '{las_path.name}'...")
        with laspy.open(las_path) as f:
            if f.header.point_count == 0:
                print(f"  - 警告: 文件 '{las_path.name}' 不包含任何点，已跳过。")
                return False

            las = f.read()
            classifications = las.classification

        # 确保数据类型为 uint8
        classifications_uint8 = np.array(classifications, dtype=np.uint8).reshape(-1,1)
        
        print(f"  - 已提取 {len(classifications_uint8)} 个点的分类信息。")
        print(f"  - 正在写入 NumPy 文件: '{npy_path.name}'...")

        # 【核心步骤变更】
        # 使用 np.save() 来创建 .npy 文件。
        # 这个函数会自动处理文件写入，并包含必要的头信息。
        np.save(npy_path, classifications_uint8)

        print(f"  - 成功: 已保存 '{npy_path.name}'.")
        return True

    except Exception as e:
        print(f"  - 错误: 处理 '{las_path.name}' 时发生错误: {e}")
        return False


def main():
    """
    批量处理的主函数。
    """
    print("--- 开始批量转换Classification到.npy文件 ---")
    start_time = time.time()

    input_p = Path(INPUT_FOLDER)
    output_p = Path(OUTPUT_FOLDER)

    if not input_p.is_dir():
        print(f"错误: 输入文件夹 '{INPUT_FOLDER}' 不存在。")
        return

    print(f"输出文件夹将被创建于: '{output_p}'")
    output_p.mkdir(parents=True, exist_ok=True)

    files_to_process = [p for p in input_p.iterdir() if p.suffix.lower() in ['.las', '.laz']]
    
    if not files_to_process:
        print("在输入文件夹中没有找到任何 .las 或 .laz 文件。")
        return
        
    print(f"找到 {len(files_to_process)} 个LAS/LAZ文件，开始转换...")
    print("-" * 20)

    total_files = len(files_to_process)
    success_count = 0

    for i, las_file_path in enumerate(files_to_process):
        print(f"[{i+1}/{total_files}] 正在处理: {las_file_path.name}")
        
        # 构建输出文件名，例如 a.las -> a.npy
        basename = las_file_path.stem
        npy_file_path = output_p / f"{basename}.npy" # <-- 后缀变更为 .npy
        
        if convert_classification_to_npy(las_file_path, npy_file_path):
            success_count += 1
        
        print("-" * 20)

    end_time = time.time()
    print("--- 批量转换完成 ---")
    print(f"总计: {total_files} 个文件。")
    print(f"成功转换: {success_count} 个文件。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == '__main__':
    
    # --- 【1. 配置区】请修改为您自己的文件夹路径 ---
    # 存放原始 .las 或 .laz 文件的文件夹
    INPUT_FOLDER = r'E:\data\WHU-Railway3D-las\urban_railway\output'

    # 用于保存输出的 .npy 文件的文件夹
    OUTPUT_FOLDER = r'E:\data\WHU-Railway3D-las\urban_railway\submission'
    
    main()
    