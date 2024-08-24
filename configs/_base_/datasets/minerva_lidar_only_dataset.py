
'''
######################################### DESCRIPTION ##########################################
#
# Copied from "michele_custom_dataset_no_images-car.py"
'''

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
#           - (TODO (1)) db_sampler is missing here, but it's optional so can also work without it. It is needed for the
#             selection of more accurate instances (i.e. to  filter them based on the difficulty and number of points in
#             the bbox). Will use it in the 3-class version.
#           - maybe need to add "MultiscaleFlipAug3D" in test_pipeline but for now let's leave it blank (TODO (2))
#           - (TODO (3)) modify data_prefix to 'training/velodyne' in dataloaders
#           - (TODO (TODO)) check why all dataloaders have the 'training/velodyne' instead of their own one
#           - Add "test_dataloader" and "test_evaluator" (straight copy form validation)
#           - (TODO (TODO)) check if can still train without the test parts!)
#           - (TODO (TODO)) check why test_dataloader has the same "val.pkl" file for the annotations...there is the test one!!
#           - Add the visualization part (straight copy from kitti)






'''
######################################### PARAMETERS ##########################################
'''

# dataset settings
dataset_type = 'MinervaLidarOnlyDataset'
data_root = 'data/minerva_polimove/'
class_names = ['Car']  # replace with your dataset class
point_cloud_range = [-80, -25, -1, 200, 25, 5]  # adjust according to your dataset
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
default_backend_args = None






'''
######################################### DB-SAMPLER ##########################################
'''

# Added for Minerva and copied from kitti+pointpillars 3d-3class
# Modified some stuff such as:  - {obvious} classes (from 3 to 1)
#                               - {obvious} info_path
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'minerva_polimove_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],                  # Try to leave it even though I don't have the parameter in
                                                    # the dataset
        filter_by_min_points=dict(Car=5)),                          ## Can be adjusted in case of need
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

# Modified to make it fit PointPillars
#       - Added Obj. Sample
#       - Removed Obj. Noise 
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4,
         backend_args=default_backend_args),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True),
    dict(type='ObjectSample', 
         db_sampler=db_sampler, 
         use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points',
               'gt_bboxes_3d',
               'gt_labels_3d'])
]
# Added augmentation (MultiScaleFlipAug3D) to make it fit PointPillars
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # replace with your point cloud data dimension
        use_dim=4),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),                                 ## Sure it can remain??
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='GlobalRotScaleTrans',
                  rot_range=[0, 0],
                  scale_ratio_range=[1., 1.],
                  translation_std=[0, 0, 0]),
             dict(type='RandomFlip3D'),
             # Experimented:    - either add both or remove both (otherwise there are boxes with no points at all)
             #                  - Kitti+Pointpillars just keeps the "PointsRangeFilter"
             #                  - I will proceed to remove both of them, so that the test is done on the whole pointcloud. If 
             #                    then the pre-processing will directly cut the point-cloud, I can cut it here too
             dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
             dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range)
         ]),
    dict(type='Pack3DDetInputs', 
         keys=['points'])                                                       ## Here only points because don't have labels!!
]
# Point-cloud filtering: will keep it because not interested in the rest
eval_pipeline = [
    dict(type='LoadPointsFromFile', 
        coord_type='LIDAR', 
        load_dim=4, 
        use_dim=4),
    dict(type='Pack3DDetInputs',
         keys=['points']),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range)
]






'''
######################################### DATALOADERS ##########################################
'''

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',                                       ## This makes the total number of epochs equal to N*n_epochs
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='minerva_polimove_infos_train.pkl',  # specify your training pkl info
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
        ann_file='minerva_polimove_infos_val.pkl',  # specify your validation pkl info
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
        ann_file='minerva_polimove_infos_val.pkl',                        
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))






'''
######################################### EVALUATORS ##########################################
'''

# Will need to modify here...
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'minerva_polimove_infos_val.pkl',  # specify your validation pkl info
    metric='bbox')
test_evaluator = val_evaluator






'''
######################################### VISUALIZATION ##########################################
'''

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')