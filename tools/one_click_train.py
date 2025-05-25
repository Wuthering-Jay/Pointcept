import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from utils.tile import process_las_files
from utils.misc import *

#################### 训练基础参数 #######################
batch_size = 2                                        # bs: total bs in all gpus
epoch = 10                                           # 训练轮数
resume = False                                          # 是否从上次训练中断处继续训练
weight_path = None                                      # 预训练模型路径，None表示不加载预训练模型
las_dir = r"D:\data\天津样例数据\细粒度"                  # 点云数据文件夹，训练用的las点云位于该文件夹下的train目录中
config_file = "configs/tj_f/semseg-pt-v2m2-0-base.py"   # 配置文件路径
save_path = "exp/tj_f/semseg-pt-v2m2-0-base"            # 训练日志与模型保存路径
ignore_labels=[] # 忽略的类别标签
num_classes = 8                                        # 训练的类别数
point_distance = 0.25                                   # 点云采样间隔，单位米，略大于点云平均距离
########################################################


train_data_dir = os.path.join(las_dir, "train")
train_npy_dir = os.path.join(las_dir, "npy/train")
label_mapping_file = os.path.join(train_npy_dir, "label_mapping.json")
label_statistics_file = os.path.join(train_npy_dir, "label_statistics.json")

# 若 las_dir/train 目录下的las文件已被处理过，则不需要再次处理，可将下面的代码注释掉
# process_las_files(
#     input_path=train_data_dir,
#     output_dir=train_npy_dir,
#     ignore_labels=ignore_labels,
#     window_size=(100,100),
#     min_points=4096,
#     max_points=65536,
#     label_count=True,
#     label_remap=True,
#     output_format="npy",
#     test_mode=False
# )

weights=extract_sorted_weights(label_statistics_file)

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()
    
def main():

    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"
    options = None

    cfg = default_config_parser(config_file, options)
    cfg.batch_size = batch_size
    cfg.epoch = epoch
    cfg.eval_epoch = epoch
    cfg.resume = resume
    cfg.weight = weight_path
    cfg.save_path = save_path
    cfg.model.criteria[0].weight = weights
    cfg.data_root = os.path.join(las_dir, "npy")
    cfg.data.train.data_root = os.path.join(las_dir, "npy")
    cfg.data.val.data_root = os.path.join(las_dir, "npy")
    cfg.data.test.data_root = os.path.join(las_dir, "npy")
    cfg.data.names=extract_label_id(label_mapping_file)
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