weight = 'exp/yn2city/semseg-pt-v2m5-3-base/model/model_last.pth'
resume = False
evaluate = True
test_only = False
seed = 42
save_path = 'exp/yn2city/semseg-pt-v2m5-3-base'
num_worker = 0
batch_size = 3
batch_size_val = None
batch_size_test = None
epoch = 50
eval_epoch = 10
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_freq = 25
empty_cache_per_epoch = True
enable_weighted_sampler = True
find_unused_parameters = False
mix_prob = 0
param_dicts = None
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
num_classes = 8
grid_size = 1.0
dataset_type = 'LasDataset'
data_root = 'D:\data\云南遥感中心数据第二批\new_disk03\tile200'
ignore_index = -1
names = [
    'ground', 'vegetation', 'building', 'bridge', 'powerline', 'vehicle',
    'wall', 'greenhouse'
]
model = dict(
    type='DefaultSegmentor',
    backbone=dict(
        type='PT-v2m5',
        in_channels=6,
        num_classes=8,
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
        grid_sizes=(3.0, 7.5, 18.75, 46.875),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend='interp'),
    criteria=[
        dict(
            type='CrossEntropyLoss',
            weight=[
                0.0076957091118362185, 0.005701996970914352,
                0.026335640307692337, 0.3123823075545282, 0.3723610310246578,
                0.1308470146197928, 0.07846983374632867, 0.06620646666424963
            ],
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
scheduler = dict(type='CosineAnnealingLR', total_steps=50)
data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        'ground', 'vegetation', 'building', 'bridge', 'powerline', 'vehicle',
        'wall', 'greenhouse'
    ],
    train=dict(
        type='LasDataset',
        split='train',
        data_root='D:\data\云南遥感中心数据第二批\new_disk03\tile200',
        transform=[
            dict(type='CentroidShift', apply_z=True),
            dict(type='NormalizeIntensity'),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=1.0,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment', 'intensity', 'is_first', 'is_last'),
                feat_keys=('coord', 'intensity', 'is_first', 'is_last'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=5),
    val=dict(
        type='LasDataset',
        split='val',
        data_root='D:\data\云南遥感中心数据第二批\new_disk03\tile200',
        transform=[
            dict(type='CentroidShift', apply_z=True),
            dict(type='NormalizeIntensity'),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=1.0,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment', 'intensity', 'is_first', 'is_last'),
                feat_keys=('coord', 'intensity', 'is_first', 'is_last'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='LasDataset',
        split='val',
        data_root='D:\data\云南遥感中心数据第二批\new_disk03\tile200',
        transform=[
            dict(type='CentroidShift', apply_z=True),
            dict(type='NormalizeIntensity')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample_Maxloop',
                grid_size=1.0,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                max_test_loops=10),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'index', 'intensity', 'is_first',
                          'is_last'),
                    feat_keys=('coord', 'intensity', 'is_first', 'is_last'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }]]),
        ignore_index=-1))
