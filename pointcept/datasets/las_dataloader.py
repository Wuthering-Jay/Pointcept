"""
LAS/LAZ Point Cloud Dataset

Author: [Your Name]
"""

import os
import json
import numpy as np
import laspy
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .defaults import DefaultDataset


@DATASETS.register_module()
class LasDataset(DefaultDataset):
    """Dataset for LAS/LAZ point cloud files"""
    
    VALID_ASSETS = [
        "coord",
        "segment",
        "intensity",
        "echo_ratio",
        "is_first",
        "is_last",
    ]
    
    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
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
                # If it's a file, assume it's a JSON file listing LAS/LAZ files
                with open(split_path) as f:
                    data_list += [
                        os.path.join(self.data_root, data) for data in json.load(f)
                    ]
            elif os.path.isdir(split_path):
                # If it's a directory, look for LAS/LAZ files
                for root, _, files in os.walk(split_path):
                    for file in files:
                        if file.lower().endswith(('.las', '.laz')):
                            data_list.append(os.path.join(root, file))
            else:
                raise ValueError(f"Invalid path: {split_path}")
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        
        # Use cached data if enabled
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        
        # Read LAS/LAZ file
        try:
            las = laspy.read(data_path)
            
            # Always extract coordinates
            data_dict["coord"] = np.vstack((las.x, las.y, las.z)).transpose()
            
            # Extract classification as segment if available
            if hasattr(las, "classification"):
                
                data_dict["segment"] = np.array(las.classification)
            
            # Extract intensity if available
            if hasattr(las, "intensity"):
                intensity = np.array(las.intensity, dtype=np.float32)
                # 标准化：减去均值，除以标准差
                intensity_mean = intensity.mean()
                intensity_std = intensity.std()
                if intensity_std > 0:
                    data_dict["intensity"] = (intensity - intensity_mean) / intensity_std
                else:
                    # 如果标准差为0，说明所有值相同，直接置为0
                    data_dict["intensity"] = np.zeros_like(intensity)
            
            # Calculate echo ratio if return_number and num_returns are available
            if hasattr(las, "return_number") and hasattr(las, "number_of_returns"):
                return_number = np.array(las.return_number, dtype=np.float32)
                number_of_returns = np.array(las.number_of_returns, dtype=np.float32)
                
                # Echo ratio
                data_dict["echo_ratio"] = return_number / (number_of_returns + 1e-6)
                
                # Is first return: 1 if return_number == 1, else -1
                data_dict["is_first"] = np.where(return_number == 1, 1, -1).astype(np.float32)
                
                # Is last return: 1 if return_number == number_of_returns, else -1
                data_dict["is_last"] = np.where(return_number == number_of_returns, 1, -1).astype(np.float32)

        except Exception as e:
            logger = get_root_logger()
            logger.error(f"Error reading {data_path}: {e}")
            # Create empty data with minimal required fields
            data_dict["coord"] = np.zeros((0, 3), dtype=np.float32)
        
        # Add metadata
        data_dict["name"] = name
        data_dict["split"] = split
        
        # If segment is not available, create default one filled with -1
        if "segment" not in data_dict:
            data_dict["segment"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        
        # Ensure segment is 1D
        data_dict["segment"] = data_dict["segment"].reshape([-1])
        
        # Add instance data (not typically available in LAS/LAZ files)
        data_dict["instance"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        
        return data_dict

    def get_data_name(self, idx):
        return os.path.splitext(os.path.basename(self.data_list[idx % len(self.data_list)]))[0]

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
    
