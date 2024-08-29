
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
point_cloud_range = [-70, -20, -2, 150, 20, 5]                  ## How to adjust? Use "tools/misc/browse_datase.py" after setting 
                                                                #  the line "PointsRangeFilter" in test_pipeline to NON-commented
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
        filter_by_difficulty=[-1],                  # Does not give problems, even though I don't have the parameter in the dataset
        filter_by_min_points=dict(Car=20)),                     # Can be used since the problem with the counting function in
                                                                # the "data creation" pipeline has been fixed
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
    # Idea: maybe if I remove these for the training but then feed it for the testing and validation, the NN is not ready to 
    # deal with the rest of the points
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
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
        load_dim=4,
        use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            # Experimented:     - Kitti+Pointpillars just keeps the "PointsRangeFilter"
            #                   - I tried to keep both, but the problem is the following. If you keep "ObjectRangeFilter" the 
            #                     transformer looks for the labels. But we did not upload them with "LoadAnnotations3D" so
            #                     it gives back an error...
            # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range)
         ]),
    # Starting to understand: should be what you pass to the actual neural network. So here just keep the points (no labels)
    dict(type='Pack3DDetInputs',                    
        keys=['points'])
]
# For now undestood that:   - the only difference with "test_pipeline" is that here the "MultiScaleFlipAug3D" is 
#                             not performed
#                           - if add the "LoadAnnotations3D" then the eval_dataloader does not work, probably since 
#                             they are not present in the ".pkl" file
eval_pipeline = [
    dict(type='LoadPointsFromFile', 
        coord_type='LIDAR', 
        load_dim=4, 
        use_dim=4),
    dict(type='Pack3DDetInputs',
        keys=['points'])
]






'''
######################################### DATALOADERS ##########################################
'''

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',                                       ## This makes the total number of epochs equal to N*n_epochs
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='minerva_polimove_infos_train.pkl',
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
        ann_file='minerva_polimove_infos_val.pkl',
        pipeline=eval_pipeline,                             ## Changed it to eval, before was test_pipeline and actually did not 
                                                            #  really make much sense. 
                                                            #  Even more so because the "_pipeline" variables are used as temp 
                                                            #  and not definitive variables.
        modality=input_modality,
        test_mode=True,                         ## If put "False" then the dataloader tries to use annotations, but they are not 
                                                #  present in the pipeline. 
                                                #  At the same time, if try to add annotations in the pipeline gives error since
                                                #  the ".pkl" file does not have them.
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

# TODO: Will need to modify here...
val_evaluator = dict(
    type='MinervaMetric',
    ann_file=data_root + 'minerva_polimove_infos_val.pkl',
    metric='bbox'
)
test_evaluator = val_evaluator






'''
######################################### VISUALIZATION ##########################################
'''

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')