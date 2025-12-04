import ctypes
import os
import time
import shutil
import tempfile
from typing import Optional, Tuple, List
from tqdm import tqdm

# ================= 配置区域 =================
# DLL 路径 (固定值)
DLL_PATH = r"libs\Release\LiDAROprationDLLEx.dll"
# ===========================================


def batch_lac_process(
    input_dir: str,
    output_dir: str,
    use_tile: bool = False,
    window_size: Tuple[float, float] = (1000.0, 1000.0),
    min_points: Optional[int] = 10000,
    max_points: Optional[int] = None,
    label_remap: bool = False,
    label_count: bool = False,
    save_sample_weight: bool = False,
    require_labels: Optional[List[int]] = None,
    use_trash_bin: bool = False,
    trash_bin_label: int = 0,
    dll_path: str = DLL_PATH
) -> dict:
    """
    批量处理 LAS 文件进行 LAC 处理
    
    Args:
        input_dir: 输入 LAS 文件夹路径
        output_dir: 输出文件夹路径
        use_tile: 是否对 LAS 文件进行 tile 分块处理（处理大文件时启用）
        window_size: tile 窗口大小，默认 (1000, 1000)
        min_points: tile 最小点数阈值，默认 10000
        max_points: tile 最大点数阈值，默认 None
        label_remap: 是否重映射标签，默认 False
        label_count: 是否统计标签，默认 False
        save_sample_weight: 是否保存样本权重，默认 False
        require_labels: 需要保留的标签列表，默认 None
        use_trash_bin: 是否启用垃圾桶机制，默认 False
        trash_bin_label: 垃圾桶标签值，默认 0
        dll_path: DLL 文件路径，默认使用配置的路径
    
    Returns:
        处理结果统计字典，包含 success_count, fail_count, failed_files, elapsed_time
    """
    result = {
        "success_count": 0,
        "fail_count": 0,
        "failed_files": [],
        "elapsed_time": 0.0
    }
    
    # 检查 DLL
    if not os.path.exists(dll_path):
        print(f"错误：找不到 DLL 文件 -> {dll_path}")
        return result

    # 加载 DLL
    try:
        lib = ctypes.CDLL(dll_path)
        lib.StartProcess.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        print(">>> DLL 加载成功")
    except Exception as e:
        print(f"DLL 加载失败: {e}")
        return result

    # 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f">>> 创建输出目录: {output_dir}")

    start_time = time.time()

    if use_tile:
        # 使用 tile 模式处理大文件
        print(">>> 启用 tile 分块处理模式")
        
        from utils.las_tile import process_las_tiles
        from utils.las_merge import merge_las_segments
        
        # 创建临时目录存放中间文件
        temp_tile_dir = tempfile.mkdtemp(prefix="lac_tile_")
        temp_lac_dir = tempfile.mkdtemp(prefix="lac_result_")
        
        try:
            # Step 1: 对输入文件进行 tile 分块
            print(f"\n>>> Step 1: Tile 分块处理...")
            process_las_tiles(
                input_path=input_dir,
                output_dir=temp_tile_dir,
                window_size=window_size,
                min_points=min_points,
                max_points=max_points,
                label_remap=label_remap,
                label_count=label_count,
                save_sample_weight=save_sample_weight,
                require_labels=require_labels,
                use_trash_bin=use_trash_bin,
                trash_bin_label=trash_bin_label
            )
            
            # Step 2: 对 tile 后的文件进行 LAC 处理
            print(f"\n>>> Step 2: LAC 处理...")
            las_files = [f for f in os.listdir(temp_tile_dir) if f.lower().endswith('.las')]
            total_files = len(las_files)
            
            if total_files == 0:
                print(">>> 未找到 tile 后的 LAS 文件。")
                return result
            
            print(f">>> 发现 {total_files} 个 tile 文件")
            
            for filename in tqdm(las_files, desc="LAC 处理", unit="file"):
                in_path = os.path.join(temp_tile_dir, filename)
                out_path = os.path.join(temp_lac_dir, filename)
                
                try:
                    b_in = in_path.encode('gbk')
                    b_out = out_path.encode('gbk')
                    lib.StartProcess(b_in, b_out)
                    result["success_count"] += 1
                except Exception as e:
                    result["fail_count"] += 1
                    result["failed_files"].append((filename, str(e)))
            
            # Step 3: 合并 LAC 处理后的文件
            print(f"\n>>> Step 3: 合并结果...")
            merge_las_segments(
                input_path=temp_lac_dir,
                output_dir=output_dir,
                label_remap_file=None
            )
            
        finally:
            # 清理临时目录
            print(f"\n>>> 清理临时文件...")
            if os.path.exists(temp_tile_dir):
                shutil.rmtree(temp_tile_dir)
            if os.path.exists(temp_lac_dir):
                shutil.rmtree(temp_lac_dir)
    else:
        # 直接处理模式
        las_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.las')]
        total_files = len(las_files)

        if total_files == 0:
            print(">>> 未找到 LAS 文件。")
            return result

        print(f">>> 发现 {total_files} 个文件")

        for filename in tqdm(las_files, desc="LAC 处理", unit="file"):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)
            
            try:
                b_in = in_path.encode('gbk')
                b_out = out_path.encode('gbk')
                lib.StartProcess(b_in, b_out)
                result["success_count"] += 1
            except Exception as e:
                result["fail_count"] += 1
                result["failed_files"].append((filename, str(e)))

    # 结束
    result["elapsed_time"] = time.time() - start_time
    
    print(f"\n>>> 全部完成！耗时: {result['elapsed_time']:.2f} 秒")
    print(f">>> 成功: {result['success_count']}, 失败: {result['fail_count']}")
    print(f">>> 结果已保存在: {output_dir}")
    
    if result["failed_files"]:
        print("\n>>> 失败文件列表:")
        for fname, err in result["failed_files"]:
            print(f"    - {fname}: {err}")
    
    return result


if __name__ == "__main__":
    # 示例用法
    INPUT_DIR = r"E:\data\梯田\output3\output_filtered\KM35.las"
    OUTPUT_DIR = r"E:\data\梯田\output3\output_filtered\KM35"
    
    batch_lac_process(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        use_tile=False,  # 启用 tile 分块处理
        window_size=(1000.0, 1000.0),  # tile 窗口大小
        min_points=1000,  # 最小点数阈值
    )