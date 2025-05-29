import os
import sys

os.chdir("..")

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch
from utils.tile import process_las_files
from utils.merge import merge_las_segments
from utils.misc import *

#################### 预测基础参数 #######################
las_dir = r"D:\data\天津样例数据\细粒度8"  # 点云数据文件夹，训练用的las点云位于该文件夹下的train目录中
output_dir = r"D:\data\天津样例数据\细粒度8\pred"  # 预测结果保存路径
weight_path = "exp/tj_f/semseg-pt-v2m2-0-base/model/model_last.pth"  # 预训练模型路径
config_file = "configs/tj_f/semseg-pt-v2m2-0-base.py"  # 配置文件路径
save_path = "exp/tj_f/semseg-pt-v2m2-0-base"  # 日志保存路径
num_classes = 5  # 训练的类别数
point_distance = 0.25  # 点云采样间隔，单位米，略大于点云平均距离
########################################################


train_data_dir = os.path.join(las_dir, "train")
test_data_dir = os.path.join(las_dir, "train")
train_npy_dir = os.path.join(las_dir, "npy/train")
test_npy_dir = os.path.join(las_dir, "npy/test")
label_mapping_file = os.path.join(train_npy_dir, "label_mapping.json")
label_statistics_file = os.path.join(train_npy_dir, "label_statistics.json")

# 若 las_dir/train 目录下的las文件已被处理过，则不需要再次处理，可将下面的代码注释掉
# process_las_files(
#     input_path=test_data_dir,
#     output_dir=test_npy_dir,
#     ignore_labels=[],
#     window_size=(100, 100),
#     min_points=4096,
#     max_points=65536/2,
#     label_count=False,
#     label_remap=False,
#     output_format="npy",
#     test_mode=True
# )

weights = extract_sorted_weights(label_statistics_file)


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"
    options = None

    cfg = default_config_parser(config_file, options)
    cfg.save_path = save_path
    cfg.weight = weight_path
    cfg.model.criteria[0].weight = weights
    cfg.data_root = os.path.join(las_dir, "npy")
    cfg.data.train.data_root = os.path.join(las_dir, "npy")
    cfg.data.val.data_root = os.path.join(las_dir, "npy")
    cfg.data.test.data_root = os.path.join(las_dir, "npy")
    cfg.data.test.split = "test"
    cfg.data.names = extract_label_id(label_mapping_file)
    cfg.model.backbone.num_classes = num_classes
    cfg.data.num_classes = num_classes
    cfg.model.backbone.grid_sizes = (
        point_distance * 3,
        point_distance * 7.5,
        point_distance * 18.75,
        point_distance * 46.875,
    )
    cfg.data.train.transform[4].grid_size = point_distance
    cfg.data.val.transform[4].grid_size = point_distance
    cfg.data.test.test_cfg.voxelize.grid_size = point_distance

    launch(
        main_worker,
        num_gpus_per_machine=num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
    merge_las_segments(
        input_path=test_npy_dir,
        output_dir=output_dir,
        input_format="npy",
        label_file="pred",
        label_remap_file=label_mapping_file,
    )
