_base_ = ['../_base_/schedules/cosine.py', '../_base_/custom_runtime_fusion.py', '../_base_/datasets/minerva_camera_lidar_dataset.py']

# model settings --> ORIGINAL
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
sparse_shape_default=[41, 1600, 1408]
#
# ------------> "sparse_shape_default" and "SparseEncoder-->output_channels" was wrong and was 
#               causing mistakes. Explanation follows
#           ------> If you put sparse_shape[z_value]=41 (it should be =40) then you get double 
#                   the output_channels in SparseEncoder (which is default=128 while second 
#                   has input=256)
#           ------> So what is needed to do is to select the right sparse_shape[z_value] and
#                   explicitate the output value of SparseEncoder to output_channels=256
#
# ------------> "sparse_shape_default" must have a specific property or will get mistakes. The
#               explanation follows 
#           ------> The dimensions y,x must be dividable by 2 for 4 times (essentially by 16).
#                   So for example 600 is not ok (600/8=75 so not further dividable by 2) while
#                   1200 is ok (1200/8=150 dividable by 2)
#           ------> This is because the pointcloud canvas is processed with a three-fold convolution by 
#                   "(make_encoder_layers/sparse_encoder.py) -> (make_sparse_convmodule/sparse_block.py)"
#                   that has stride=2. This means that the canvas's dimensions are diminished of 2^3=8
#                   along all of these directions.
#           ------> When the canvas is then processed by SECONDFPN it is further divided by 2, which can 
#                   cause some problems when the re-upsampling happens (example: normal version has y=75
#                   and downsampled has y=38, but then 38*2=76)
#
# ------------> For the z, this problem is actually absorbed by the collapsing along the z direction. But
#               since the code is not really well done, it is desirable that the number of voxels in the
#               z-direction is "slightly" superior to be divided by 8.
#               For example, if 64 is the desired number of voxels (can be divided by 8) then make it a 
#               bit higher (like 65 or 66).
#           ------> Sometimes there may be little issues, like if z-grid=32 then must make it
#                   34 or it doesn't work (not even with 33)
#
# model settings --> MODIFIED
voxel_size = [0.05, 0.05, 0.2]
point_cloud_range = [0, -28, -2, 120, 28, 4.8]
sparse_shape_default=[
    int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2]),     # z dimension
    int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),     # y dimension
    int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])]     # x dimension

model = dict(
    type='DynamicMVXFasterRCNN',
    save_losses_on_file = False,
    losses_file_destination_path = "/home/michele/code/michele_mmdet3d/demo/losses_log.json",
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1)),
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # make the image features more stable numerically to avoid loss nan
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        fusion_layer=dict(
            type='PointFusion',
            img_channels=256,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=False)),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=128,
        output_channels=256,
        sparse_shape=sparse_shape_default,
        order=('conv', 'norm', 'act'),

        # NOTE: Modified version, comment to use standard values --> Default values are 
        #       in SparseEncoder for the optional arguments
        # base_channels=32,
        # encoder_channels=((32,),
        #                   (64, 64, 64),
        #                   (128, 128, 128),
        #                   (128, 128, 128))

    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -30, -1, 120, 30, -1],
            ],
            sizes=[
                [5.0, 2.0, 1.5]
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)))

train_cfg = dict(max_epochs=300, val_interval=100)

optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# You may need to download the model first is the network is unstable
# load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'  # noqa

resume = False

work_dir = './work_dirs/MINERVA_mvxet'
