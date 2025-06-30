import numpy as np
import plyfile
import laspy
import os
from pathlib import Path

def convert_ply_to_las(ply_path, las_path):
    """
    将包含 x, y, z, intensity, 和 label 的 PLY 文件转换为 LAS 文件。
    (这是我们之前的核心转换函数)

    Args:
        ply_path (str): 输入的 PLY 文件路径。
        las_path (str): 输出的 LAS 文件路径。
    """
    try:
        # 1. 读取 PLY 文件
        print(f"--- 正在处理: {Path(ply_path).name} ---")
        plydata = plyfile.PlyData.read(ply_path)
        points = plydata['vertex']

        # 从 PLY 数据中提取字段
        x = points['x']
        y = points['y']
        z = points['z']
        intensity = points['intensity']
        labels = points['class'].astype(np.uint8)

        # 2. 创建 LAS 文件头
        header = laspy.LasHeader(version="1.2", point_format=1)
        
        # 计算并设置坐标的偏移和缩放
        header.offsets = [np.min(x), np.min(y), np.min(z)]
        header.scales = [0.01, 0.01, 0.01] # 可根据需要调整精度

        # 3. 填充 LAS 数据
        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z
        las.intensity = intensity
        las.classification = labels

        # 4. 写入 LAS 文件
        las.write(las_path)
        print(f"成功转换并保存到: {las_path}")

    except FileNotFoundError:
        print(f"错误: 文件未找到 - {ply_path}")
    except Exception as e:
        print(f"处理文件 {Path(ply_path).name} 时发生错误: {e}")
        print("请确保该PLY文件包含 'x', 'y', 'z', 'intensity', 'label' 字段。")
    finally:
        print("-" * 30)


def batch_convert_ply_to_las(input_folder, output_folder):
    """
    批量转换一个文件夹内所有的 .ply 文件到 .las 文件。

    Args:
        input_folder (str): 包含 .ply 文件的输入文件夹路径。
        output_folder (str): 用于保存 .las 文件的输出文件夹路径。
    """
    # 将输入和输出路径转换为 Path 对象，方便操作
    in_path = Path(input_folder)
    out_path = Path(output_folder)

    # 1. 检查输入文件夹是否存在
    if not in_path.is_dir():
        print(f"错误: 输入文件夹 '{input_folder}' 不存在或不是一个文件夹。")
        return

    # 2. 检查并创建输出文件夹
    print(f"输出文件夹将被设置为: {output_folder}")
    out_path.mkdir(parents=True, exist_ok=True)

    # 3. 查找所有 .ply 文件
    ply_files = list(in_path.glob("*.ply"))

    if not ply_files:
        print(f"在文件夹 '{input_folder}' 中没有找到任何 .ply 文件。")
        return

    print(f"找到 {len(ply_files)} 个 .ply 文件。开始批量转换...")
    
    # 4. 遍历并转换每个文件
    for ply_file_path in ply_files:
        # 构建输出文件的完整路径
        # 例如，将 'input/data1.ply' 转换为 'output/data1.las'
        output_las_path = out_path / ply_file_path.with_suffix(".las").name
        
        # 调用单个文件转换函数
        convert_ply_to_las(ply_file_path, output_las_path)

    print("所有文件转换完成！")


# --- 使用示例 ---
if __name__ == '__main__':
    # *** 请在这里修改您的文件夹路径 ***
    
    # 包含您的 .ply 文件的文件夹
    input_directory = r'D:\WHU-Railway3D\test_set\rural_railways' 
    
    # 您希望保存转换后的 .las 文件的文件夹
    output_directory = r'D:\WHU-Railway3D-las\rural_railway\test'

    # 执行批量转换
    batch_convert_ply_to_las(input_directory, output_directory)