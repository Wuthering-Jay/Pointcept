import os
import numpy as np
import laspy
from plyfile import PlyData
import time

def ply_to_las(ply_path, las_path):
    """
    核心转换函数：将单个 PLY 转换为 LAS
    """
    try:
        # 1. 读取 PLY
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        # 2. 准备 LAS Header
        # point_format=2 (支持 RGB, 无 GPS 时间)
        header = laspy.LasHeader(point_format=2, version="1.2")
        
        # 3. 设置 Scale 和 Offset (这对 SensatUrban 这种大坐标数据至关重要)
        # Scale: 0.001 (毫米级精度)
        header.scales = [0.001, 0.001, 0.001]
        # Offset: 使用当前文件的最小坐标作为原点，防止精度丢失
        header.offsets = [np.min(vertex['x']), np.min(vertex['y']), np.min(vertex['z'])]

        # 4. 创建 LAS 对象
        las = laspy.LasData(header)

        # 5. 填充坐标
        las.x = vertex['x']
        las.y = vertex['y']
        las.z = vertex['z']

        # 6. 填充颜色 (RGB 8bit -> 16bit)
        # 检查字段名是否存在 (red/green/blue 或 r/g/b)
        names = vertex.data.dtype.names
        if 'red' in names:
            las.red = vertex['red'].astype(np.uint16) * 256
            las.green = vertex['green'].astype(np.uint16) * 256
            las.blue = vertex['blue'].astype(np.uint16) * 256
        elif 'r' in names: # 处理简写的情况
            las.red = vertex['r'].astype(np.uint16) * 256
            las.green = vertex['g'].astype(np.uint16) * 256
            las.blue = vertex['b'].astype(np.uint16) * 256

        # 7. 填充类别 (Class)
        # SensatUrban 通常是 'class' 或 'label'
        if 'class' in names:
            las.classification = vertex['class'].astype(np.uint8)
        elif 'label' in names:
            las.classification = vertex['label'].astype(np.uint8)
        
        # 8. 写入文件
        las.write(las_path)
        return True, f"点数: {len(las.x)}"

    except Exception as e:
        return False, str(e)

def batch_process(target_folder):
    """
    批量处理逻辑
    """
    # 检查文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"错误: 文件夹 '{target_folder}' 不存在。")
        return

    # 获取所有 .ply 文件 (不包含子文件夹)
    # 使用列表推导式过滤文件
    files = [f for f in os.listdir(target_folder) 
             if f.lower().endswith('.ply') and os.path.isfile(os.path.join(target_folder, f))]
    
    total_files = len(files)
    
    if total_files == 0:
        print("未找到 .ply 文件。")
        return

    print(f"--- 开始处理，共找到 {total_files} 个文件 ---")
    print(f"目标文件夹: {target_folder}\n")

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, filename in enumerate(files):
        src_path = os.path.join(target_folder, filename)
        # 输出文件名：原文件名.las
        dst_path = os.path.join(target_folder, os.path.splitext(filename)[0] + ".las")

        print(f"[{i+1}/{total_files}] 正在转换: {filename} ...", end="", flush=True)
        
        # 执行转换
        status, msg = ply_to_las(src_path, dst_path)
        
        if status:
            print(f" 完成 ({msg})")
            success_count += 1
        else:
            print(f" 失败! 原因: {msg}")
            fail_count += 1

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- 处理结束 ---")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")

# --- 配置区域 ---
if __name__ == "__main__":
    # 在这里修改为你的 ply 文件夹路径
    # Windows 路径示例: r"D:\Datasets\SensatUrban\train"
    # Linux 路径示例: "/home/user/data/sensaturban"
    
    target_directory = r"E:\data\SensatUrban" 
    
    batch_process(target_directory)