_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = True 
evaluate = True
batch_size = 16  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
empty_cache_freq = 50
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= True
save_path = "exp/dales/semseg-pnext-v1m2-1-base"
weight = "exp/dales/semseg-pnext-v1m2-1-base/model/model_last.pth"
num_classes = 8
grid_size = 0.2

# dataset settings
dataset_type = "LasDataset"
data_root = r"E:\data\dales_las\tiles"

ignore_index = -1
names = [
    "ground",
    "vegetation",
    "cars",
    "trucks",
    "power lines",
    "fences",
    "poles",
    "buildings",
]


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PNext-m2",
        in_channels=4,
        num_classes=num_classes,
        patch_embed_depth=1,
        patch_embed_channels=16,
        patch_embed_neighbours=16,
        enc_depths=(1, 1, 1, 1),
        enc_channels=(32, 64, 128, 256),
        enc_neighbours=(32, 32, 32, 32),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(16, 32, 64, 128),
        dec_neighbours=(32, 32, 32, 32),
        grid_sizes=(
            3 * grid_size,
            3 * 2.5 * grid_size,
            3 * 2.5 * 2.5 * grid_size,
            3 * 2.5 * 2.5 * 2.5 * grid_size,
        ),  # x3, x2.5, x2.5, x2.5
        drop_path_rate=0.3,
        unpool_backend="interp",  # map / interp
    ),
    # fmt: off
    criteria=[
        dict(type="CrossEntropyLoss",
             weight=[
0.029863364538482734,
0.03398225904062671,
0.12243503810237331,
0.18499369495513007,
0.18097568901650252,
0.14633794302484154,
0.2577365862459248,
0.04367542507611825
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
                keys=("coord", "segment", "echo_ratio"),
                feat_keys=("coord", "echo_ratio"),
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
                keys=("coord", "segment", "echo_ratio"),
                feat_keys=("coord", "echo_ratio"),
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
                    keys=("coord", "index", "echo_ratio"),
                    feat_keys=("coord", "echo_ratio"),
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
