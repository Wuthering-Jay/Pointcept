# import laspy
# import numpy as np
# from pathlib import Path
# import time


# # -------------------------------------------------

# def convert_classification_to_npy(las_path, npy_path):
#     """
#     读取一个LAS/LAZ文件，提取其classification数据，
#     并将其保存为一个NumPy格式的.npy文件（数据类型为uint8）。

#     Args:
#         las_path (Path): 输入的LAS/LAZ文件路径对象。
#         npy_path (Path): 输出的.npy文件路径对象。

#     Returns:
#         bool: 如果成功则返回True，否则返回False。
#     """
#     try:
#         print(f"  - 正在读取: '{las_path.name}'...")
#         with laspy.open(las_path) as f:
#             if f.header.point_count == 0:
#                 print(f"  - 警告: 文件 '{las_path.name}' 不包含任何点，已跳过。")
#                 return False

#             las = f.read()
#             classifications = las.classification

#         # 确保数据类型为 uint8
#         classifications_uint8 = np.array(classifications, dtype=np.uint8).reshape(-1,1)
        
#         print(f"  - 已提取 {len(classifications_uint8)} 个点的分类信息。")
#         print(f"  - 正在写入 NumPy 文件: '{npy_path.name}'...")

#         # 【核心步骤变更】
#         # 使用 np.save() 来创建 .npy 文件。
#         # 这个函数会自动处理文件写入，并包含必要的头信息。
#         np.save(npy_path, classifications_uint8)

#         print(f"  - 成功: 已保存 '{npy_path.name}'.")
#         return True

#     except Exception as e:
#         print(f"  - 错误: 处理 '{las_path.name}' 时发生错误: {e}")
#         return False


# def main():
#     """
#     批量处理的主函数。
#     """
#     print("--- 开始批量转换Classification到.npy文件 ---")
#     start_time = time.time()

#     input_p = Path(INPUT_FOLDER)
#     output_p = Path(OUTPUT_FOLDER)

#     if not input_p.is_dir():
#         print(f"错误: 输入文件夹 '{INPUT_FOLDER}' 不存在。")
#         return

#     print(f"输出文件夹将被创建于: '{output_p}'")
#     output_p.mkdir(parents=True, exist_ok=True)

#     files_to_process = [p for p in input_p.iterdir() if p.suffix.lower() in ['.las', '.laz']]
    
#     if not files_to_process:
#         print("在输入文件夹中没有找到任何 .las 或 .laz 文件。")
#         return
        
#     print(f"找到 {len(files_to_process)} 个LAS/LAZ文件，开始转换...")
#     print("-" * 20)

#     total_files = len(files_to_process)
#     success_count = 0

#     for i, las_file_path in enumerate(files_to_process):
#         print(f"[{i+1}/{total_files}] 正在处理: {las_file_path.name}")
        
#         # 构建输出文件名，例如 a.las -> a.npy
#         basename = las_file_path.stem
#         npy_file_path = output_p / f"{basename}.npy" # <-- 后缀变更为 .npy
        
#         if convert_classification_to_npy(las_file_path, npy_file_path):
#             success_count += 1
        
#         print("-" * 20)

#     end_time = time.time()
#     print("--- 批量转换完成 ---")
#     print(f"总计: {total_files} 个文件。")
#     print(f"成功转换: {success_count} 个文件。")
#     print(f"总耗时: {end_time - start_time:.2f} 秒。")


# if __name__ == '__main__':
    
#     # --- 【1. 配置区】请修改为您自己的文件夹路径 ---
#     # 存放原始 .las 或 .laz 文件的文件夹
#     INPUT_FOLDER = r'E:\data\WHU-Railway3D-las\urban_railway\output'

#     # 用于保存输出的 .npy 文件的文件夹
#     OUTPUT_FOLDER = r'E:\data\WHU-Railway3D-las\urban_railway\submission'
    
#     main()
    
import os
import numpy as np
import laspy
from tqdm import tqdm

def create_submission_files(las_folder, output_folder):
    """
    从 .las 文件的 'classification' 字段提取标签，
    并强制保存为一维数组的 .npy 文件。
    """
    if not os.path.isdir(las_folder):
        print(f"错误：找不到原始LAS文件夹 '{las_folder}'。")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"输出将保存到 '{output_folder}' 文件夹。")

    las_files = [f for f in os.listdir(las_folder) if f.lower().endswith('.las')]
    if not las_files:
        print(f"警告：在 '{las_folder}' 文件夹中没有找到 .las 文件。")
        return

    for las_file_name in tqdm(las_files, desc="正在生成NPY文件"):
        las_file_path = os.path.join(las_folder, las_file_name)
        
        try:
            las_file = laspy.read(las_file_path)
            
            # 提取 classification 字段并转换为 NumPy 数组
            labels_np = np.array(las_file.classification, dtype=np.uint8)
            
            # 构建输出路径
            base_name = os.path.splitext(las_file_name)[0]
            npy_file_path = os.path.join(output_folder, f"{base_name}.npy")
            
            # --- 修改之处 ---
            # 在保存前使用 .flatten() 强制确保数组为一维
            np.save(npy_file_path, labels_np.flatten())

        except Exception as e:
            print(f"\n处理文件 {las_file_name} 时出错: {e}")

    print("\n所有文件处理完成！")
    print("已强制将所有输出保存为一维数组。")


if __name__ == '__main__':
    # --- 请配置您的文件夹路径 ---
    
    # 您原始的 .las 文件所在的文件夹
    original_las_folder = r'F:\WHU-Railways3D\rural_railway\result' # 示例路径，请修改
    
    # 您希望保存 .npy 文件的文件夹
    submission_folder = r'F:\WHU-Railways3D\rural_railway\result\submission'
    
    # --- 运行生成程序 ---
    create_submission_files(original_las_folder, submission_folder)