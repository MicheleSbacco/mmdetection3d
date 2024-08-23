# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset






'''
######################################### LIDAR ONLY ##########################################
'''

@DATASETS.register_module()
class MinervaLidarOnlyDataset(Det3DDataset):
    
    # Define the classes we are interested in (similar to file "update_infos_to_v2.py", same issue with the class 'Extra')
    METAINFO = {
        'classes': ('Car', 'Extra')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')



    # This whole function was deleted because the docs said that everything is the same as Det3DDataset, except for the 
    # fact that here the "plane" option was added. 
    # But we don't use it so who actually gives a fuck? 
    #  
    # def parse_data_info(self, info: dict) -> dict:



    def parse_ann_info(self, info: dict) -> dict:
        
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)
        ann_info = self._remove_dontcare(ann_info)
        
        # This line is commented because this is the dataset without images
        # lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        
        # ATTENTION: Here below the type of box is defined. This part is really important to make the overall dataset work.
        #       - More information about the conventions on frames etc. can be found in the files that
        #         introduce the classes "CameraInstance3DBoxes"(mmdet3d/structures/bbox3d/cam_box3d.py) and 
        #         "LiDARInstance3DBoxes"(mmdet3d/structures/bbox3d/lidar_box3d.py)
        #       - Probably need to make sure that the type indicated here is the same as the type indicated
        #         in the field "box_type_3d" of the dataset
        #       - What KITTI did was to create Camera type boxes and then convert them with the appropriate
        #         transformation and appropriate function into LiDAR boxes  
        #       - TODO: Make a choice that works and explain it
        #       - TODO: Also check the "load_type" 
        gt_bboxes_3d = CameraInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info






'''
######################################### CAMERA AND LIDAR ##########################################
'''

@DATASETS.register_module()
class MinervaCameraLidarDataset(Det3DDataset):
    pass