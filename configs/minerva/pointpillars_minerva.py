_base_ = [
    '../_base_/datasets/minerva_lidar_only_dataset.py',
    '../_base_/schedules/cyclic_minerva_pointpillars.py', 
    '../_base_/default_runtime.py'
]






'''
######################################### GENERAL DESCRIPTION, PARAMETERS ##########################################
'''
# How this file works:  - The dataset, training scheduler, and runtime are inherited from files in "_base_/..." 
#                       - The NN model as a whole is described here






# # dataset settings
# dataset_type = 'MinervaLidarOnlyDataset'
# data_root = 'data/minerva_polimove/'
# class_names = ['Car']  # replace with your dataset class
# input_modality = dict(use_lidar=True, use_camera=False)
# metainfo = dict(classes=class_names)
# default_backend_args = None






point_cloud_range = [-80, -25, -1, 200, 25, 5]              ## Make sure is the same as reference value in "_base_/datasets/..."
voxel_size = [0.16, 0.16, (point_cloud_range[5]-point_cloud_range[2])]







'''
######################################### MODEL OF NEURAL NETWORK ##########################################
'''

###################################################### PROCESSING OF DATA

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6]                           ## Just one because it's just one class
            ],
            sizes=[[5., 2., 1.5]],                                      ## Adapted to dimension of car
            rotations=[-0.34, 0, 0.34],                                         ## Adapted to type of circuit: can be left just
                                                                                #  one with 0° of rotation 
            reshape_out=False),
        diff_rad_by_sin=True,
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

###################################################### TRAINING AND TESTING CONFIG

    train_cfg=dict(
        assigner=[
            dict(                                           ## Tuning for validation of bboxes
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
