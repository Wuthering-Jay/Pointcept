_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = True
evaluate = True
batch_size = 2  # bs: total bs in all gpus
num_worker = 0
mix_prob = 0
empty_cache = False
empty_cache_freq = 100
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= False
save_path = "exp/nl/semseg-deeplanet-v1m1-0-20260516"
weight = "exp/nl/semseg-deeplanet-v1m1-0-20260516/model/model_last.pth"
# weight = None
num_classes = 2
grid_size = 0.25

# dataset settings
dataset_type = "LasDataset"
data_root = r"E:\data\铁二院\第二批\优化\nl\tile100"

ignore_index = -1
names = [
    "non-ground",
    "ground",
]
class_weight = [1.0,1.0]


# model settings
model = dict(
    type="DeepLASegmentor",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="DeepLANet-v2",
        in_channels=5,
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
            9 * grid_size,
            27 * grid_size,
            81 * grid_size,
        ),  # x3, x2.5, x2.5, x2.5
        drop_path_rate=0.3,
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
            # dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment","is_first","is_last", "grid_coord"),
                feat_keys=("coord","is_first","is_last",),
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
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="CentroidShift", apply_z=True),
            # dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment","is_first","is_last", "grid_coord"),
                feat_keys=("coord","is_first","is_last",),
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
            # dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
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
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index","is_first","is_last", "grid_coord"),
                    feat_keys=("coord","is_first","is_last",),
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
