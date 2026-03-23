_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = False
evaluate = True
batch_size = 2  # bs: total bs in all gpus
num_worker = 2
mix_prob = 4
empty_cache = False
empty_cache_freq = 100
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= True
save_path = "exp/yn2city/semseg-deeplanet-v2-20260323"
# weight = "exp/yn/semseg-pt-v2m5-1-base/model/model_last.pth"
num_classes = 8
grid_size = 0.75

# dataset settings
dataset_type = "LasDataset"
data_root = r"E:\data\云南遥感中心\精修城区（侧立面、桥梁）\tile"

ignore_index = -1
names = [
    "ground",
    "vegetation",
    "building",
    "bridge",
    "powerline",
    "vehicle",
    "wall",
    "greenhouse"
]

class_weight = [
    0.013817545610175072,
    0.014948150592024706,
    0.023757059743400432,
    0.14248988080367028,
    0.5757914131469801,
    0.1064141385417331,
    0.04740625042980346,
    0.07537556113221294
]


# model settings
model = dict(
    type="DeepLASegmentor",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="DeepLANet-v2",
        in_channels=6,
        patch_embed_depth=1,
        patch_embed_channels=32,
        patch_embed_neighbours=16,
        enc_depths=(10, 10, 30, 10),
        enc_channels=(64, 128, 256, 512),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(64, 128, 256, 512),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(
            3 * grid_size,
            7.5 * grid_size,
            18.75 * grid_size,
            45.875 * grid_size,
        ),  # x3, x2.5, x2.5, x2.5
        drop_path_rate=0.2,
        enable_checkpoint=False,
        unpool_backend="interp",
        # 深层网络稳定性优化
        enable_deep_supervision=True,   # 启用 HDS (混合深监督)
        enable_layer_scale=True,        # 启用 LayerScale
        layer_scale_init_value=1e-5,    # LayerScale 初始值
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1, weight=class_weight),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    # 辅助损失配置 (Hybrid Deep Supervision)
    # loss_weight=0.4 相当于原来的 hds_alpha
    aux_criteria=[
        dict(type="CrossEntropyLoss", loss_weight=0.4, ignore_index=-1, weight=class_weight),
    ],
    # 辅助头通道数，需与 enc_channels 对应
    aux_channels=(64, 128, 256, 512),
    aux_dropout=0.1,
    # 各 stage 的辅助损失权重比例 (越深的层权重越大)
    aux_weights=(0.1, 0.2, 0.3, 0.4),
)

# scheduler settings
epoch = 50
eval_epoch = 10
optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-4)
clip_grad = 5.0

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
        split="val",
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
