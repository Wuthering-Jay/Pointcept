_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = True
evaluate = True
batch_size = 4  # bs: total bs in all gpus
num_worker = 4
mix_prob = 0
empty_cache = False
empty_cache_freq = 100
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= False
save_path = "exp/nl/semseg-pt-v2m5-1-20260414"
# weight = "exp/nl/semseg-pt-v2m5-0-base/model/model_last.pth"
weight = None
num_classes = 2
grid_size = 0.25

# dataset settings
dataset_type = "LasDataset"
data_root = r"D:\Data\铁二院优化\nl\tile"

ignore_index = -1
names = [
    "non-ground",
    "ground",
]


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m5",
        in_channels=6,
        num_classes=num_classes,
        patch_embed_depth=1,
        patch_embed_channels=24,
        patch_embed_groups=6,
        patch_embed_neighbours=16,
        enc_depths=(1, 2, 1),
        enc_channels=(48, 96, 192),
        enc_groups=(6, 12, 24),
        enc_neighbours=(24, 24, 24),
        dec_depths=(1, 1, 1),
        dec_channels=(24, 48, 96),
        dec_groups=(4, 6, 12),
        dec_neighbours=(24, 24, 24),
        grid_sizes=(
            3 * grid_size,
            9 * grid_size,
            27 * grid_size,
            # 2.34375 * grid_size * 20,
        ),  # x3, x2.5, x2.5, x2.5
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="interp",  # map / interp
    ),
    # fmt: off
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             weight=[1.0, 0.5],
             ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        # dict(type="FocalLoss", gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1),
    ],
    # fmt: on
)

# scheduler settings
epoch = 25
eval_epoch = 5
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
            dict(type="CentroidShift", apply_z=True),
            dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
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
            # dict(type="StandardNormalize", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment","intensity","is_first","is_last",),
                feat_keys=("coord","intensity","is_first","is_last",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CentroidShift", apply_z=True),
            dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
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
            # dict(type="StandardNormalize", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment","intensity","is_first","is_last",),
                feat_keys=("coord","intensity","is_first","is_last",),
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
            dict(type="CentroidShift", apply_z=True),
            dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample_Maxloop",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                max_test_loops=20
            ),
            crop=None,
            post_transform=[
                # dict(type="PointClip",point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),),
                # dict(type="StandardNormalize", apply_z=True),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index","intensity","is_first","is_last",),
                    feat_keys=("coord","intensity","is_first","is_last",),
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
