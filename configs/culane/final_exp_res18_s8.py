from configs.culane.test_common_s8 import *
"""
    config file of the small version of lanesegnet for culane
"""
# global settings
dataset_type = 'CulaneDataset'
data_root = "xxx/CULane/"
test_mode = False
fpn_layer_num = 3                            # check
fpn_down_scale = [8,16,32]                   # check
mask_down_scale = 8                          # check
hm_down_scale = 8                            # check
line_width = 3
radius = 2  # gaussian circle radius
root_radius = 4
vaniehsd_radius = 8
joint_nums = 1                               # check
joint_weights = [1, 0.4, 0.2]                # check
sample_per_lane = [41, 21, 11]               # check
dcn_point_num = [7, 5, 3]                    # check
sample_gt_points = [41, 21, 11]              # check
loss_weights = dict(center=0.0,
                    point=1.0,
                    error=1.0,
                    offset=0.5,
                    aux=0.2,
                    rela_offset=0.5
                )                            # check
use_smooth = False                           # check
deconv_before = False                        # check
dcn_only_cls = True                          # check
point_scale = False                          # check
deconv_layer = [True, False, False]          # check
nms_thr = 2
num_lane_classes = 1
batch_size = 32                              # check
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)
ori_scale = (1640, 590)  # for culane
crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)
train_cfg = dict(out_scale=mask_down_scale)
test_cfg = dict(out_scale=mask_down_scale)
assigner_cfg = dict(
    init=dict(
        assigner=dict(type='LaneAssigner')),
    refine=dict(
        assigner=dict(type='LaneAssigner'))
)
# model settings
model = dict(
    type='LaneSegNet',
    pretrained=None, #'torchvision://resnet18'
    train_cfg=train_cfg,
    test_cfg=test_cfg,
    num_classes=num_lane_classes,
    sample_gt_points=sample_gt_points,
    use_smooth=use_smooth,
    point_scale=point_scale,
    use_geb=True,
    geb_cfg=dict(
        in_channels=64
    ),
    backbone=dict(
        type='ResNet',
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='DeformFPN',
        in_channels=[128, 256, 512],
        out_channels=64,
        dcn_point_num=dcn_point_num,
        deconv_layer=deconv_layer,
        deconv_before=deconv_before,
        trans_idx=-1,
        dcn_only_cls=dcn_only_cls,
        trans_cfg=dict(
            in_dim=512,
            attn_in_dims=[512, 64],
            attn_out_dims=[64, 64],
            strides=[1, 1],
            ratios=[4, 4],
            pos_shape=(1, 10, 25),
        ),
    ),
    head=dict(
        type='HeadFast',
        heads=dict(hm=num_lane_classes),
        in_channels=64,
        branch_in_channels=64,
        num_classes=num_lane_classes,
        hm_idx=0,
        joint_nums=joint_nums,

    ),
    loss=dict(type='LaneLossAggress'),
    loss_weights=loss_weights
)

train_compose = dict(bboxes=False, keypoints=True, masks=True)

# data pipeline settings
train_al_pipeline = [
    dict(type='Compose', params=train_compose),
    # Geometric transforms (applied to both image and label)
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
    dict(
        type='RandomResizedCrop',
        height=img_scale[1],
        width=img_scale[0],
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

# Appearance transforms (applied to image only)
train_al_pipeline_img_only = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0),
        ],
        p=0.7),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
]

val_al_pipeline = [
    dict(type='Compose', params=train_compose),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type='albumentation', pipelines=train_al_pipeline),#    dict(type='albumentation', pipelines=train_al_pipeline_img_only),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLanePoints',
        fpn_layer_num=fpn_layer_num,
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        max_mask_sample=5,
        line_width=line_width,
        radius=radius,
        root_radius=root_radius,
        vanished_radius=vaniehsd_radius,
        joint_nums=joint_nums,
        joint_weights=joint_weights,
        sample_per_lane=sample_per_lane,
        fpn_down_scale=fpn_down_scale,
        keys=['img', 'gt_cpts_hm', 'gt_kpts_hm', 'int_offset', 'pts_offset',
              'gt_masks', *[f'lane_points_l{i}' for i in range(fpn_layer_num)],
              'offset_mask', 'offset_mask_weight', 'gt_vp_hm', 'rela_offset_maps'],
        meta_keys=[
            'filename', 'sub_img_name',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points'
        ]),
]

val_pipeline = [
    dict(type='albumentation', pipelines=val_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLanePoints',
        fpn_layer_num=fpn_layer_num,
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        max_mask_sample=5,
        line_width=line_width,
        radius=radius,
        root_radius=root_radius,
        vanished_radius=vaniehsd_radius,
        joint_nums=joint_nums,
        joint_weights=joint_weights,
        sample_per_lane=sample_per_lane,
        fpn_down_scale=fpn_down_scale,
        keys=['img', 'gt_cpts_hm', 'gt_kpts_hm', 'int_offset', 'pts_offset',
              'gt_masks', *[f'lane_points_l{i}' for i in range(fpn_layer_num)],
              'offset_mask', 'offset_mask_weight', 'gt_vp_hm', 'rela_offset_maps'],
        meta_keys=[
            'filename', 'sub_img_name',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points'
        ]),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/train.txt',
        seg_label_list=data_root + '/list/train_gt.txt',
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/test.txt',
        seg_label_list=data_root + '/list/test_gt.txt',
        pipeline=val_pipeline,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/test.txt',
        seg_label_list=data_root + '/list/test_gt.txt',
        test_suffix='.jpg',
        pipeline=val_pipeline,
        test_mode=True,
    ))

# optimizer
# optimizer = dict(type='Adam', lr=2.5e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[50, 100])

# runtime settings
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

total_epochs = 200
device_ids = "0,1"
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tusimple/small'
load_from = None
resume_from = None
workflow = [('train', 500), ('val', 1)]

from configs.culane.common import *