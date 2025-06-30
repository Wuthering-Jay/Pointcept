import os
import glob
import numpy as np
import laspy
from sklearn.decomposition import PCA

def rotate_road_pcd_to_y_axis(input_las_path: str, output_las_path: str):
    """
    (这是我们之前的核心函数，无需修改)
    Rotates a road point cloud from a LAS file so that its principal direction
    aligns with the Y-axis. The rotation is performed only in the XY plane.
    This version is compatible with laspy 2.x and higher.

    Args:
        input_las_path (str): Path to the input LAS file.
        output_las_path (str): Path to save the rotated LAS file.
    """
    print(f"  正在读取文件: {os.path.basename(input_las_path)}")
    try:
        with laspy.open(input_las_path) as f:
            las = f.read()

        points_xy = np.vstack((las.X, las.Y)).transpose()

        print("  正在执行 PCA...")
        pca = PCA(n_components=2)
        pca.fit(points_xy)
        principal_vector = pca.components_[0]
        
        angle_road = np.arctan2(principal_vector[1], principal_vector[0])
        angle_target = np.pi / 2
        rotation_angle = angle_target - angle_road

        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        
        new_x = points_xy[:, 0] * cos_theta - points_xy[:, 1] * sin_theta
        new_y = points_xy[:, 0] * sin_theta + points_xy[:, 1] * cos_theta
        
        print("  正在应用旋转并保存...")
        new_las = laspy.create(point_format=las.header.point_format,
                               file_version=las.header.version)
        new_las.header = las.header
        new_las.points = las.points.copy()
        
        new_las.X = new_x
        new_las.Y = new_y
        new_las.Z = las.Z

        new_las.write(output_las_path)
        print(f"  成功保存到: {os.path.basename(output_las_path)}")

    except Exception as e:
        print(f"  处理文件 {os.path.basename(input_las_path)} 时发生错误: {e}")


def batch_rotate_las_in_folder(input_folder: str, output_folder: str):
    """
    (这是新增的批量处理函数)
    Finds all .las files in the input folder, rotates them using PCA,
    and saves the results to the output folder.

    Args:
        input_folder (str): The path to the folder containing input .las files.
        output_folder (str): The path to the folder where rotated files will be saved.
    """
    print(f"开始批量处理...")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")

    # 1. 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 2. 查找所有 .las 文件
    # 使用 glob 可以方便地匹配所有 .las 文件
    search_path = os.path.join(input_folder, '*.las')
    las_files = glob.glob(search_path)

    if not las_files:
        print("\n在输入文件夹中未找到任何 .las 文件。请检查路径。")
        return

    print(f"\n找到了 {len(las_files)} 个 .las 文件，准备开始处理。")
    
    # 3. 循环处理每一个文件
    for i, input_path in enumerate(las_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename)
        
        print(f"\n--- [{i+1}/{len(las_files)}] 正在处理: {filename} ---")
        
        # 调用核心函数处理单个文件
        rotate_road_pcd_to_y_axis(input_path, output_path)

    print("\n所有文件处理完毕！")


if __name__ == '__main__':
    # --- 配置区域 ---
    # 请将这里替换为您的输入文件夹路径
    INPUT_FOLDER_PATH = r'E:\data\WHU-Railway3D-las\plateau_railway\test'
    
    # 请将这里替换为您希望保存结果的输出文件夹路径
    OUTPUT_FOLDER_PATH = r'E:\data\WHU-Railway3D-las\plateau_railway\test_rotated'

    # --- 执行批量处理 ---
    batch_rotate_las_in_folder(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH)