_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
resume = True
evaluate = True
batch_size = 5  # bs: total bs in all gpus
num_worker = 5
mix_prob = 0
empty_cache = False
empty_cache_freq = 100
empty_cache_per_epoch = True
enable_amp = True
enable_weighted_sampler= False
save_path = "exp/nl/semseg-litept-v1m1-0-20260516"
weight = "exp/nl/semseg-litept-v1m1-0-20260516/model/model_last.pth"
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


# model settings
model = dict(
    type='DefaultSegmentorV2',
    num_classes=2,
    backbone_out_channels=72,
    backbone=dict(
        type='LitePT-v1m1',
        in_channels=5,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1, weight = [1,0.3]),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 10

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=2e-3)
scheduler = dict(
    type="CosineAnnealingLR",
    total_steps=epoch,
)
param_dicts = [dict(keyword="block", lr=1e-4)] # example: [dict(keyword="block", lr_scale=0.1)]


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
        split="test",
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
