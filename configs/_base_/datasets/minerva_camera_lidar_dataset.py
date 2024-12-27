
'''
######################################### DESCRIPTION ##########################################
# 
### See "minerva_lidar_only_dataset.py" to see more information about dataloaders etc.
# 
# 
### Copied from 
#       - "minerva_lidar_only_dataset.py"
#       - "kitti-3d-car_MULTIVIEW.py"
'''






'''
######################################### PARAMETERS ##########################################
'''

# dataset settings
dataset_type = 'MinervaCameraLidarDataset'
data_root = 'data/minerva_polimove/'
class_names = ['Car']
point_cloud_range = [-70, -20, -2, 150, 20, 5]                  ## How to adjust? Use "tools/misc/browse_datase.py" after setting 
                                                                #  the line "PointsRangeFilter" in test_pipeline to NON-commented
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)
default_backend_args = None






'''
######################################### DB-SAMPLER ##########################################
'''

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'minerva_polimove_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=default_backend_args),
    backend_args=default_backend_args)






'''
######################################### PIPELINES ##########################################
'''

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=default_backend_args),
    
    # Added (OBVIOUSLY) for the visualization of images
    dict(type='LoadImageFromFile', backend_args=default_backend_args),
    
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',

        # Added some keys related to images (img, gt_bboxes, gt_labels)
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ])]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=default_backend_args),
    
    # Added (OBVIOUSLY) for the visualization of images
    dict(type='LoadImageFromFile', backend_args=default_backend_args),
    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img'
        ])]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=default_backend_args),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img'
        ])]






'''
######################################### DATALOADERS ##########################################
'''

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            modality=input_modality,
            ann_file='minerva_polimove_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced', img="training/image_2"),
            pipeline=train_pipeline,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=default_backend_args
        )))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced', img="training/image_2"),
        ann_file='minerva_polimove_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=default_backend_args
    ))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced', img="training/image_2"),
        ann_file='minerva_polimove_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=default_backend_args))






'''
######################################### EVALUATORS ##########################################
'''

val_evaluator = dict(
    ann_file='data/minerva_polimove/minerva_polimove_infos_val.pkl',
    metric='bbox',
    lidar_path_prefix = '/home/michele/code/michele_mmdet3d/',                                              # Needs update!!!
    model_path = '/home/michele/code/michele_mmdet3d/configs/minerva/MINERVA_mvxnet.py',                    # Needs update!!!
    last_chkpt_file_path = '/home/michele/code/michele_mmdet3d/work_dirs/MINERVA_mvxnet/last_checkpoint',   # Needs update!!!
    save_losses_on_file = True,
    losses_file_destination_path = "/home/michele/code/michele_mmdet3d/demo/losses_log.json",
    reduced_x_limit = [0, 80],
    delete_checkpoints = False,
    checkpoints_folder = '/home/michele/code/michele_mmdet3d/work_dirs/MINERVA_mvxnet/',
    save_checkpoints_one_every_n = 10,
    type='MinervaMetricFusion')
test_evaluator = val_evaluator






'''
######################################### VISUALIZATION ##########################################
'''

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
