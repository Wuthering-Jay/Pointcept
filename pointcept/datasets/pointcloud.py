import os
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class PointCloudDataset(DefaultDataset):
    def __inti__(self, **kwargs):
        super(PointCloudDataset, self).__init__(**kwargs)
