"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import json
from re import split

import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
# 默认数据加载
class DefaultDataset(Dataset):
    # 数据集的有效属性列表
    VALID_ASSETS = [
        "coord", # 坐标
        "echo_ratio", # 回波穿透比
        "color", # 颜色
        "normal", # 法线
        "strength", # 强度
        "segment", # 语义分割
        "instance", # 实例分割
        "pose", # 位姿
    ]

    def __init__(
        self,
        split="train", # 数据集划分，这里仅用于数据查找，即位于data_root下的哪个子文件中
        data_root="data/dataset", # 数据集根目录
        transform=None, # 数据增强
        test_mode=False, # 是否测试模式，该模式下，loop固定为1，启用test数据增强
        test_cfg=None, # 测试配置
        cache=False, # 是否使用缓存数据
        ignore_index=-1, # 忽略的语义索引
        loop=1, # 数据集循环次数
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None)
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop, os.path.basename(self.data_root), split
            )
        )

    # 获取数据列表
    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            split_path = os.path.join(self.data_root, split)
            if os.path.isfile(split_path):
                with open(split_path) as f:
                    data_list += [
                        os.path.join(self.data_root, data) for data in json.load(f)
                    ]
            elif os.path.isdir(split_path):
                # 获取指定目录下的所有条目
                all_entries = os.listdir(split_path)
                # 遍历所有条目，筛选出子文件夹
                for entry in all_entries:
                    entry_path = os.path.join(split_path, entry)
                    if os.path.isdir(entry_path):
                        data_list.append(entry_path)
            else:
                raise ValueError(f"Invalid path: {split_path}")
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)] # 获取数据路径
        name = self.get_data_name(idx) # 获取数据名称
        split = self.get_split_name(idx) # 获取数据划分
        # 读取缓存数据，依赖SharedArray库
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        # data_path 下应当包含 coord.npy/color.npy/normal.npy 等
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split
        # coord、color、normal 为 float32
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)
            
        if "echo_ratio" in data_dict.keys():
            data_dict["echo_ratio"] = data_dict["echo_ratio"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)
            
        # segment、instance 为 int32，若不存在用-1填充
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def get_split_name(self, idx):
        return os.path.basename(
            os.path.dirname(self.data_list[idx % len(self.data_list)])
        )

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        # 将 sengment 和 name 移动到 result_dict 中
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        # 若存在 origin_segment 和 inverse，则将其移动到 result_dict 中
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        # 数据增强，使用深拷贝保证原始数据不被修改
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            # 体素化处理
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            # 后处理
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
# 数据集拼接加载器，用于多数据集联合训练
class ConcatDataset(Dataset):
    def __init__(self, datasets, loop=1):
        super(ConcatDataset, self).__init__()
        self.datasets = [build_dataset(dataset) for dataset in datasets]
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in the concat set.".format(
                len(self.data_list), self.loop
            )
        )

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i]), dtype=int) * i,
                    np.arange(len(self.datasets[i])),
                )
            )
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

