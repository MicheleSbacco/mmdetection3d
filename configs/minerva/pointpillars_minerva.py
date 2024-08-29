_base_ = [
    '../_base_/datasets/minerva_lidar_only_dataset.py',
    '../_base_/schedules/cyclic_minerva_pointpillars.py', 
    '../_base_/custom_runtime.py'
]






'''
######################################### GENERAL DESCRIPTION, PARAMETERS ##########################################
'''
# How this file works:  - The dataset, training scheduler, and runtime are inherited from files in "_base_/..." 
#                       - The NN model as a whole is described here

point_cloud_range = [-70, -20, -2, 150, 20, 5]              ## Make sure is the same as reference value in "_base_/datasets/..."
voxel_size = [0.1, 0.1, 7]
grid_output_size = [
    int(  (point_cloud_range[3]-point_cloud_range[0])  /  voxel_size[0]),
    int(  (point_cloud_range[4]-point_cloud_range[1])  /  voxel_size[1])
]
anchor_range = [-70, -20, -2.5, 150, 20, 2.5]






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
            max_num_points=64,                      ## Was originally 32
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
        type='PointPillarsScatter', in_channels=64, output_shape=grid_output_size),
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
                anchor_range                            ## Just one because it's just one class
            ],
            sizes=[[5., 2., 1.5]],                                      ## Adapted to dimension of car
            rotations=[0, 1.57],                                         ## Adapted to type of circuit: can be left just
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
            type='mmdet.SmoothL1Loss', 
            beta=1.0 / 9.0, 
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', 
            use_sigmoid=False,
            loss_weight=0.2)),

###################################################### TRAINING AND TESTING CONFIG

    train_cfg=dict(
        assigner=[
            dict(                                           ## Tuning for positive/negative score of bboxes (iou = intersection over union)
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
    
    # ATTENTION: This filter has the precedence over the one in the Visualization Hook. 
    # The one in the hook is like a filter after the filter
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        # Parameter "score_thr" defines the minumum score that a bbox needs, in order to 
        # be shown in the results of the inference
        score_thr=5e-7,
        min_bbox_size=0,
        nms_pre=100,
        # Number of "top bboxes" among which the "score_thr" is filtered
        max_num=4))