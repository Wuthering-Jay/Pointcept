_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = True 
evaluate = True
batch_size = 6  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
empty_cache_freq = 50
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= True
save_path = "exp/railway_plateau/semseg-pnext-v1m1-0-base"
weight = "exp/railway_plateau/semseg-pnext-v1m1-0-base/model/model_last.pth"
num_classes = 8
grid_size = 0.05

# dataset settings
dataset_type = "PointCloudDataset"
data_root = r"D:\WHU-Railway3D-las\plateau_railway\npy"

ignore_index = -1
names = [
    "rail",
    "track bed",
    "lines",
    "fences",
    "poles",
    "buildings",
    "ground",
    "others",
]


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PNext-m1",
        in_channels=4,
        num_classes=num_classes,
        enc_blocks=[1, 1, 1, 1, 1],
        dec_blocks=[1, 1, 1, 1, 1],
        planes=[16, 32, 64, 128, 256],
        stride=[1, 4, 4, 4, 4],
        nsample=[16, 32, 32, 32, 32],
    ),
    # fmt: off
    criteria=[
        dict(type="CrossEntropyLoss",
             weight=[
0.03920678546814069,
0.024261880216581024,
0.2337494840831869,
0.07944271990683648,
0.25958049626666896,
0.14305886634749432,
0.016501570435559786,
0.20419819727553176
                 ],
             loss_weight=1.0,
             ignore_index=-1),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        dict(type="FocalLoss", gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1),
    ],
    # fmt: on
)

# scheduler settings
epoch = 100
eval_epoch = 20
optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-4)


scheduler = dict(
    type="CosineAnnealingLR",
    total_steps=epoch,
)


data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/12, 1/12], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/12, 1/12], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            # dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            # dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            # dict(type="StandardNormalize", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "intensity"),
                feat_keys=("coord", "intensity"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/12, 1/12], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/12, 1/12], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            # dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            # dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            # dict(type="StandardNormalize", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "intensity"),
                feat_keys=("coord", "intensity"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample_Maxloop",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                max_test_loops=10
            ),
            crop=None,
            post_transform=[
                # dict(type="PointClip",point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),),
                # dict(type="StandardNormalize", apply_z=True),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index", "intensity"),
                    feat_keys=("coord", "intensity"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
