# Procedure:
#   - Started from the suggestions in the "Docs" --> "Advanced Guides" --> "Customization" --> "Customize Datasets"
#   - Compared to kitti and to the "Docs" --> "User Guides" --> "Train and Test" --> "Learn about Configs"
#       - General idea
#           - Dataloaders are a PyTorch entity needed by the NNs during training, validation and testing
#           - Dataloaders need pipelines, that are used as intermediate variables to simplify the code
#           - There are also "val_evaluator" and "test_evaluator" that indicate the evaluation metrics. Usually the test
#             is the same as val but can be modified (see "Learn about Configs"). These evaluators need a pipeline as
#             well, which is "eval_pipeline" of course.
#             To indicate the metrics, the metric configs are used. Can find them under "mmdet3d/evaluation/metrics"
#           - There can also be some visualization information. See kitti dataset or "Visualization" docs
#           - So at the end we need:
#               - Pipeline: train and test (for dataloaders), eval (for evaluators)
#               - Dataloader: train, val and test (they usually use the same pipeline)
#               - Evaluator: val and test
#               - Visualizer: just one standard (can use "vis_backends" as an intermediate variable)
#       Modifications:
#           - db_sampler is missing here, but it's optional so for now let's leave it blank (TODO (1))
#           - maybe need to add "MultiscaleFlipAug3D" but for now let's leave it blank (TODO (2))
#           - (TODO (3)) modify data_prefix to 'training/velodyne' in dataloaders
#           - (TODO (TODO)) check why all dataloaders have the 'training/velodyne' instead of their own one
#           - Add "test_dataloader" and "test_evaluator" (straight copy form validation)
#           - (TODO (TODO)) check if can still train without the test parts!)
#           - (TODO (TODO)) check why test_dataloader has the same "val.pkl" file for the annotations...there is the test one!!
#           - Add the visualization part (straight copy from kitti)



# dataset settings
dataset_type = 'MicheleCustomDatasetNoImages'
data_root = 'data/michele_custom/'
class_names = ['Car']  # replace with your dataset class
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # adjust according to your dataset
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)



## TODO (1): Missing db_sampler



#-------------------------------------PIPELINES---------------------------------------------------------


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # replace with your point cloud data dimension
        use_dim=4),  # replace with the actual dimension used in training and inference
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True),
    
    

    ## TODO (1): Missing object sample
    

    
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # replace with your point cloud data dimension
        use_dim=4),
    
    
    
    ## TODO (2): Missing Aumentation
    
    
    
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
eval_pipeline = [
    dict(
        type='LoadPointsFromFile', 
        coord_type='LIDAR', 
        load_dim=4, 
        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
]


#-------------------------------------DATALOADERS---------------------------------------------------------


train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='michele_custom_infos_train.pkl',  # specify your training pkl info
            data_prefix=dict(pts='training/velodyne'),      ## TODO (3)
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),      ## TODO (3)
        ann_file='michele_custom_infos_val.pkl',  # specify your validation pkl info
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),      ## TODO (3)
        ann_file='michele_custom_infos_val.pkl',                        
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))


#-------------------------------------EVALUATORS---------------------------------------------------------


val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'michele_custom_infos_val.pkl',  # specify your validation pkl info
    metric='bbox')
test_evaluator = val_evaluator


#-------------------------------------VISUALIZATION---------------------------------------------------------


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')