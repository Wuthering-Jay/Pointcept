_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = False
evaluate = True
batch_size = 48  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
empty_cache_freq = 20
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= True
save_path = "exp/yn/semseg-pt-v2m5-1-base"
# weight = "exp/yn/semseg-pt-v2m5-1-base/model/model_last.pth"
num_classes = 7
grid_size = 2

# dataset settings
dataset_type = "PointCloudDataset"
data_root = r"E:\data\云南遥感中心数据\npy"

ignore_index = -1
names = [
    "ground",
    "vegetation",
    "building",
    "water noise",
    "power line",
    "vehicle",
    "wall",
]


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m5",
        in_channels=3,
        num_classes=num_classes,
        patch_embed_depth=1,
        patch_embed_channels=24,
        patch_embed_groups=6,
        patch_embed_neighbours=16,
        enc_depths=(1, 1, 2, 1),
        enc_channels=(48, 96, 192, 256),
        enc_groups=(6, 12, 24, 32),
        enc_neighbours=(24, 24, 24, 24),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(24, 48, 96, 192),
        dec_groups=(4, 6, 12, 24),
        dec_neighbours=(24, 24, 24, 24),
        grid_sizes=(
            0.15 * grid_size * 20,
            0.375 * grid_size * 20,
            0.9375 * grid_size * 20,
            2.34375 * grid_size * 20,
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
             weight=[
                 0.004400392062962055,
                 0.004886941518634558,
                 0.016574600711464882,
                 0.07332837581634521,
                 0.5889480710029602,
                 0.25070253014564514,
                 0.06115901097655296
                 ],
             loss_weight=1.0,
             ignore_index=-1),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        dict(type="FocalLoss", gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1),
    ],
    # fmt: on
)

# scheduler settings
epoch = 30
eval_epoch = 10
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
                keys=("coord", "segment"),
                feat_keys=("coord",),
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
                keys=("coord", "segment"),
                feat_keys=("coord",),
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
                    keys=("coord", "index"),
                    feat_keys=("coord",),
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
