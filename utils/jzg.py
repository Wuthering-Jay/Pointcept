import os
from pathlib import Path
from tile import process_las_files

window_size=(50., 50.)
min_points=4096
max_points=65536
ignore_labels = []
label_remap=True
label_count=True
output_format="npy"
save_echo_ratio=True
save_color=False
save_intensity=False
save_normal=False
save_index=False
test_mode=False

def process_all_folders(base_dir):
    """
    处理所有1-49文件夹
    :param base_dir: jzg文件夹的基础路径 (如 "D:/data/jzg")
    """
    base_path = Path(base_dir)
    
    for i in range(1, 50):
        folder = base_path / str(i)
        
        # 处理train文件夹
        train_input = folder / "train" / f"{i}.las"
        train_output = folder / "npy" / "train"
        if train_input.exists():
            process_las_files(
                input_path=train_input,
                output_dir=train_output,
                window_size=window_size,
                min_points=min_points,
                max_points=max_points,
                ignore_labels=ignore_labels,
                label_remap=label_remap,
                label_count=label_count,
                output_format=output_format,
                save_echo_ratio=save_echo_ratio,
                save_color=save_color,
                save_intensity=save_intensity,
                save_normal=save_normal,
                save_index=save_index,
                test_mode=test_mode
            )
        
        # 处理test文件夹
        test_input = folder / "test" / f"{i}.las"
        test_output = folder / "npy" / "test"
        if test_input.exists():
            process_las_files(
                input_path=test_input,
                output_dir=test_output,
                window_size=window_size,
                min_points=min_points,
                max_points=max_points,
                ignore_labels=ignore_labels,
                label_remap=label_remap,
                label_count=label_count,
                output_format=output_format,
                save_echo_ratio=save_echo_ratio,
                save_color=save_color,
                save_intensity=save_intensity,
                save_normal=save_normal,
                save_index=save_index,
                test_mode=test_mode
            )

if __name__ == "__main__":
    # 设置你的jzg文件夹路径
    jzg_base_dir = r"D:\data\jzg"  # 请修改为你的实际路径
    
    # 确保基础路径存在
    if not os.path.exists(jzg_base_dir):
        raise FileNotFoundError(f"目录 {jzg_base_dir} 不存在")
    
    print("开始处理所有LAS文件...")
    process_all_folders(jzg_base_dir)
    print("所有文件处理完成!")
