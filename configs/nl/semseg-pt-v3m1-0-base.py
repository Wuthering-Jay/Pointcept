_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = True
evaluate = True
batch_size = 2  # bs: total bs in all gpus
num_worker = 2
mix_prob = 0
empty_cache = False
empty_cache_freq = 100
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= False
save_path = "exp/nl/semseg-pt-v3m1-0-20260518"
weight = "exp/nl/semseg-pt-v3m1-0-20260518/model/model_last.pth"
# weight = None
num_classes = 2
grid_size = 0.25

# dataset settings
dataset_type = "LasDataset"
data_root = r"E:\data\铁二院\第二批\优化迭代\nl\tile100"

ignore_index = -1
names = [
    "non-ground",
    "ground",
]


model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=32,
    backbone=dict(
        type="PT-v3m1",
        in_channels=5,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(1, 1, 1, 3, 1),
        enc_channels=(16, 32, 64, 128, 256),
        enc_num_head=(1, 2, 4, 8, 16),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(32, 32, 64, 128),
        dec_num_head=(2, 2, 4, 8),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Dales"),
    ),
    # fmt: off
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             weight=[1.0, 1.0],
             ignore_index=-1),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        # dict(type="FocalLoss", gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1),
    ],
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
