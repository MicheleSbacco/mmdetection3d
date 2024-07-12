# Copyright (c) OpenMMLab. All rights reserved.
"""Convert the annotation pkl to the standard format in OpenMMLab V2.0.

Example:
    python tools/dataset_converters/update_infos_to_v2.py
        --dataset kitti
        --pkl-path ./data/kitti/kitti_infos_train.pkl
        --out-dir ./kitti_v2/
"""

import argparse
import copy
import time
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes

from mmdet3d.datasets.convert_utils import (convert_annos,
                                            get_kitti_style_2d_boxes,
                                            get_nuscenes_2d_boxes)
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmdet3d.structures import points_cam2img


def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        unaligned_bbox_3d=None)
    return instance


def get_empty_multicamera_instances(camera_types):

    cam_instance = dict()
    for cam_type in camera_types:
        cam_instance[cam_type] = None
    return cam_instance


def get_empty_lidar_points():
    lidar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of LiDAR data file.
        lidar_path=None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        lidar2ego=None,
    )
    return lidar_points


def get_empty_radar_points():
    radar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of RADAR data file.
        radar_path=None,
        # Transformation matrix from lidar to
        # ego-vehicle with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        radar2ego=None,
    )
    return radar_points


def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info


def get_single_image_sweep(camera_types):
    single_image_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None)
    # (dict): Information of images captured by multiple cameras
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep


def get_single_lidar_sweep():
    single_lidar_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None,
        # (dict): Information of images captured by multiple cameras
        lidar_points=get_empty_lidar_points())
    return single_lidar_sweep


def get_empty_standard_data_info(
        camera_types=['CAM0', 'CAM1', 'CAM2', 'CAM3', 'CAM4']):

    data_info = dict(
        # (str): Sample id of the frame.
        sample_idx=None,
        # (str, optional): '000010'
        token=None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        radar_points=get_empty_radar_points(),
        # (list[dict], optional): Image sweeps data.
        image_sweeps=[],
        lidar_sweeps=[],
        instances=[],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        instances_ignore=[],
        # (str, optional): Path of semantic labels for each point.
        pts_semantic_mask_path=None,
        # (str, optional): Path of instance labels for each point.
        pts_instance_mask_path=None)
    return data_info


def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag


def generate_kitti_camera_instances(ori_info_dict):

    cam_key = 'CAM2'
    empty_camera_instances = get_empty_multicamera_instances([cam_key])
    annos = copy.deepcopy(ori_info_dict['annos'])
    ann_infos = get_kitti_style_2d_boxes(
        ori_info_dict, occluded=[0, 1, 2, 3], annos=annos)
    empty_camera_instances[cam_key] = ann_infos

    return empty_camera_instances



# Copied function: from "update_kitti_infos"
# Peculiarities:
#   - Takes a boolean "use_images" as input, to remove the modifications linked to images
#   - Additionally has a boolean "keep_calib" that, if TRUE, keeps the "calib" elements even if "use_images" is FALSE
def update_michele_custom_infos(pkl_path, out_dir, use_images=True):
    
    # My additional boolean for testing
    keep_calib = True                                           ##  Apparently needs to be true, otherwise the "create_gt_database" 
                                                                #   gives back an error. (deep functions in mmengine)

    # Just some warning prints
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)

    # IMPORTANT: these are the classes that will be considered as "relevant". The other ones will be set to "-1"
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc'),
    }

    # Two lists are created:
    #   - "data_list"       =   the instances in the "original" format
    #   - "converted_list"  =   the instances in the "MMDet" format
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []

    # A for loop that at each step:
    #   1. Takes an instance form the list "data_list" in original format
    #   2. Modifies it
    #   3. Appends it to the list "converted_list" in MMDet format
    for ori_info_dict in mmengine.track_iter_progress(data_list):

        # Take a dictionary with all the needed keys, but all empty
        temp_data_info = get_empty_standard_data_info()

        # Assign the "plane" if present in original instance
        if 'plane' in ori_info_dict:
            temp_data_info['plane'] = ori_info_dict['plane']

        # Assign the instance index                                                     ##  Needed to modify it so that it grabs the info 
                                                                                        #   from the point cloud in case images are removed
        pointer = ori_info_dict.get('image', None)
        if pointer is not None:
            temp_data_info['sample_idx'] = ori_info_dict['image']['image_idx']
        else:
            temp_data_info['sample_idx'] = ori_info_dict['point_cloud']['pc_idx']

        # Assign the projection matrices
        if use_images or keep_calib:                                                                      ## Booleans
            temp_data_info['images']['CAM0']['cam2img'] = ori_info_dict['calib'][
                'P0'].tolist()
            temp_data_info['images']['CAM1']['cam2img'] = ori_info_dict['calib'][
                'P1'].tolist()
            temp_data_info['images']['CAM2']['cam2img'] = ori_info_dict['calib'][
                'P2'].tolist()
            temp_data_info['images']['CAM3']['cam2img'] = ori_info_dict['calib'][
                'P3'].tolist()

        # Assign the image path and shape. Assign the lidar path and number of points.
        if use_images:                                                                                              ## Booleans
            temp_data_info['images']['CAM2']['img_path'] = Path(ori_info_dict['image']['image_path']).name
            h, w = ori_info_dict['image']['image_shape']
            temp_data_info['images']['CAM2']['height'] = h
            temp_data_info['images']['CAM2']['width'] = w
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict['point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(ori_info_dict['point_cloud']['velodyne_path']).name

        # Assign other "calib" infos
        if use_images or keep_calib:                                                                ## Booleans
            rect = ori_info_dict['calib']['R0_rect'].astype(np.float32)
            Trv2c = ori_info_dict['calib']['Tr_velo_to_cam'].astype(np.float32)
            lidar2cam = rect @ Trv2c
            temp_data_info['images']['CAM2']['lidar2cam'] = lidar2cam.tolist()
            temp_data_info['lidar_points']['Tr_velo_to_cam'] = Trv2c.tolist()
            
            temp_data_info['images']['CAM0']['lidar2img'] = (
                ori_info_dict['calib']['P0'] @ lidar2cam).tolist()
            temp_data_info['images']['CAM1']['lidar2img'] = (
                ori_info_dict['calib']['P1'] @ lidar2cam).tolist()
            temp_data_info['images']['CAM2']['lidar2img'] = (
                ori_info_dict['calib']['P2'] @ lidar2cam).tolist()
            temp_data_info['images']['CAM3']['lidar2img'] = (
                ori_info_dict['calib']['P3'] @ lidar2cam).tolist()
            cam2img = ori_info_dict['calib']['P2']

        # for potential usage
        if use_images or keep_calib:                                                                        ## Booleans
            temp_data_info['images']['R0_rect'] = ori_info_dict['calib'][
                'R0_rect'].astype(np.float32).tolist()
            temp_data_info['lidar_points']['Tr_imu_to_velo'] = ori_info_dict[
                'calib']['Tr_imu_to_velo'].astype(np.float32).tolist()

        # For loop:
        #   - goes through the instances inside the annotations one by one: a loop(instances in annos) in the loop(instances in dictionary)
        #   - each instance is appended to the "instance_list"
        #   - iteratively constructs a SET of the instances that need to be ignored   --->   The set does NOT have repetitions!!!
        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            num_instances = len(anns['name'])
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        anns['name'][instance_id])
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    empty_instance['bbox_label'] = -1

                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                loc = anns['location'][instance_id]
                dims = anns['dimensions'][instance_id]
                rots = anns['rotation_y'][:, None][instance_id]

                dst = np.array([0.5, 0.5, 0.5])
                src = np.array([0.5, 1.0, 0.5])

                center_3d = loc + dims * (dst - src)
                if use_images or keep_calib:                                                            ## Booleans
                    center_2d = points_cam2img(center_3d.reshape([1, 3]), cam2img, with_depth=True)
                    center_2d = center_2d.squeeze().tolist()
                    empty_instance['center_2d'] = center_2d[:2]
                    empty_instance['depth'] = center_2d[2]

                gt_bboxes_3d = np.concatenate([loc, dims, rots]).tolist()
                empty_instance['bbox_3d'] = gt_bboxes_3d
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                empty_instance['truncated'] = anns['truncated'][
                    instance_id].tolist()
                empty_instance['occluded'] = anns['occluded'][
                    instance_id].tolist()
                empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
                empty_instance['score'] = anns['score'][instance_id].tolist()
                empty_instance['index'] = anns['index'][instance_id].tolist()
                empty_instance['group_id'] = anns['group_ids'][
                    instance_id].tolist()
                empty_instance['difficulty'] = anns['difficulty'][
                    instance_id].tolist()
                empty_instance['num_lidar_pts'] = anns['num_points_in_gt'][
                    instance_id].tolist()
                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
            if use_images:                                                                          ## Booleans
                cam_instances = generate_kitti_camera_instances(ori_info_dict)
                temp_data_info['cam_instances'] = cam_instances
        
        #   1. Remove the unused keys
        #   2. Append the updated and "cleared" dictionary to the "converted_list"
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # The actual final file is NOT only the updated dictionary.
    # It is a composed dictionary that has metainfos with:
    #   - The categories, for the classes
    #   - The name of the dataset (TODO: Useless??)
    #   - The information about the version (TODO: Useless??)
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'kitti'                                                       ## TODO: Modify the "kitti" name here, and eventually other "kitti" stuff (also sub-functions)
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')



## Comments directly in "tools/creata_data.py"
def update_kitti_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc'),
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info()

        if 'plane' in ori_info_dict:
            temp_data_info['plane'] = ori_info_dict['plane']

        temp_data_info['sample_idx'] = ori_info_dict['image']['image_idx']

        temp_data_info['images']['CAM0']['cam2img'] = ori_info_dict['calib'][
            'P0'].tolist()
        temp_data_info['images']['CAM1']['cam2img'] = ori_info_dict['calib'][
            'P1'].tolist()
        temp_data_info['images']['CAM2']['cam2img'] = ori_info_dict['calib'][
            'P2'].tolist()
        temp_data_info['images']['CAM3']['cam2img'] = ori_info_dict['calib'][
            'P3'].tolist()

        temp_data_info['images']['CAM2']['img_path'] = Path(
            ori_info_dict['image']['image_path']).name
        h, w = ori_info_dict['image']['image_shape']
        temp_data_info['images']['CAM2']['height'] = h
        temp_data_info['images']['CAM2']['width'] = w
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['point_cloud']['velodyne_path']).name

        rect = ori_info_dict['calib']['R0_rect'].astype(np.float32)
        Trv2c = ori_info_dict['calib']['Tr_velo_to_cam'].astype(np.float32)
        lidar2cam = rect @ Trv2c
        temp_data_info['images']['CAM2']['lidar2cam'] = lidar2cam.tolist()
        temp_data_info['images']['CAM0']['lidar2img'] = (
            ori_info_dict['calib']['P0'] @ lidar2cam).tolist()
        temp_data_info['images']['CAM1']['lidar2img'] = (
            ori_info_dict['calib']['P1'] @ lidar2cam).tolist()
        temp_data_info['images']['CAM2']['lidar2img'] = (
            ori_info_dict['calib']['P2'] @ lidar2cam).tolist()
        temp_data_info['images']['CAM3']['lidar2img'] = (
            ori_info_dict['calib']['P3'] @ lidar2cam).tolist()

        temp_data_info['lidar_points']['Tr_velo_to_cam'] = Trv2c.tolist()

        # for potential usage
        temp_data_info['images']['R0_rect'] = ori_info_dict['calib'][
            'R0_rect'].astype(np.float32).tolist()
        temp_data_info['lidar_points']['Tr_imu_to_velo'] = ori_info_dict[
            'calib']['Tr_imu_to_velo'].astype(np.float32).tolist()

        cam2img = ori_info_dict['calib']['P2']

        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            num_instances = len(anns['name'])
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        anns['name'][instance_id])
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    empty_instance['bbox_label'] = -1

                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                loc = anns['location'][instance_id]
                dims = anns['dimensions'][instance_id]
                rots = anns['rotation_y'][:, None][instance_id]

                dst = np.array([0.5, 0.5, 0.5])
                src = np.array([0.5, 1.0, 0.5])

                center_3d = loc + dims * (dst - src)
                center_2d = points_cam2img(
                    center_3d.reshape([1, 3]), cam2img, with_depth=True)
                center_2d = center_2d.squeeze().tolist()
                empty_instance['center_2d'] = center_2d[:2]
                empty_instance['depth'] = center_2d[2]

                gt_bboxes_3d = np.concatenate([loc, dims, rots]).tolist()
                empty_instance['bbox_3d'] = gt_bboxes_3d
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                empty_instance['truncated'] = anns['truncated'][
                    instance_id].tolist()
                empty_instance['occluded'] = anns['occluded'][
                    instance_id].tolist()
                empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
                empty_instance['score'] = anns['score'][instance_id].tolist()
                empty_instance['index'] = anns['index'][instance_id].tolist()
                empty_instance['group_id'] = anns['group_ids'][
                    instance_id].tolist()
                empty_instance['difficulty'] = anns['difficulty'][
                    instance_id].tolist()
                empty_instance['num_lidar_pts'] = anns['num_points_in_gt'][
                    instance_id].tolist()
                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
            cam_instances = generate_kitti_camera_instances(ori_info_dict)
            temp_data_info['cam_instances'] = cam_instances
        
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'kitti'
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_nuscenes_infos(pkl_path, out_dir):
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
    }
    nusc = NuScenes(
        version=data_list['metadata']['version'],
        dataroot='./data/nuscenes',
        verbose=True)

    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(
            mmengine.track_iter_progress(data_list['infos'])):
        temp_data_info = get_empty_standard_data_info(
            camera_types=camera_types)
        temp_data_info['sample_idx'] = i
        temp_data_info['token'] = ori_info_dict['token']
        temp_data_info['ego2global'] = convert_quaternion_to_matrix(
            ori_info_dict['ego2global_rotation'],
            ori_info_dict['ego2global_translation'])
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict.get(
            'num_features', 5)
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['lidar_path']).name
        temp_data_info['lidar_points'][
            'lidar2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['lidar2ego_rotation'],
                ori_info_dict['lidar2ego_translation'])
        # bc-breaking: Timestamp has divided 1e6 in pkl infos.
        temp_data_info['timestamp'] = ori_info_dict['timestamp'] / 1e6
        for ori_sweep in ori_info_dict['sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            temp_lidar_sweep['lidar_points'][
                'lidar2ego'] = convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation'])
            temp_lidar_sweep['ego2global'] = convert_quaternion_to_matrix(
                ori_sweep['ego2global_rotation'],
                ori_sweep['ego2global_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_sweep['sensor2lidar_rotation']
            trans = ori_sweep['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            temp_lidar_sweep['lidar_points'][
                'lidar2sensor'] = lidar2sensor.astype(np.float32).tolist()
            temp_lidar_sweep['timestamp'] = ori_sweep['timestamp'] / 1e6
            temp_lidar_sweep['lidar_points']['lidar_path'] = ori_sweep[
                'data_path']
            temp_lidar_sweep['sample_data_token'] = ori_sweep[
                'sample_data_token']
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)
        temp_data_info['images'] = {}
        for cam in ori_info_dict['cams']:
            empty_img_info = get_empty_img_info()
            empty_img_info['img_path'] = Path(
                ori_info_dict['cams'][cam]['data_path']).name
            empty_img_info['cam2img'] = ori_info_dict['cams'][cam][
                'cam_intrinsic'].tolist()
            empty_img_info['sample_data_token'] = ori_info_dict['cams'][cam][
                'sample_data_token']
            # bc-breaking: Timestamp has divided 1e6 in pkl infos.
            empty_img_info[
                'timestamp'] = ori_info_dict['cams'][cam]['timestamp'] / 1e6
            empty_img_info['cam2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['cams'][cam]['sensor2ego_rotation'],
                ori_info_dict['cams'][cam]['sensor2ego_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_info_dict['cams'][cam]['sensor2lidar_rotation']
            trans = ori_info_dict['cams'][cam]['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            empty_img_info['lidar2cam'] = lidar2sensor.astype(
                np.float32).tolist()
            temp_data_info['images'][cam] = empty_img_info
        ignore_class_name = set()
        if 'gt_boxes' in ori_info_dict:
            num_instances = ori_info_dict['gt_boxes'].shape[0]
            for i in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox_3d'] = ori_info_dict['gt_boxes'][
                    i, :].tolist()
                if ori_info_dict['gt_names'][i] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        ori_info_dict['gt_names'][i])
                else:
                    ignore_class_name.add(ori_info_dict['gt_names'][i])
                    empty_instance['bbox_label'] = -1
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['velocity'] = ori_info_dict['gt_velocity'][
                    i, :].tolist()
                empty_instance['num_lidar_pts'] = ori_info_dict[
                    'num_lidar_pts'][i]
                empty_instance['num_radar_pts'] = ori_info_dict[
                    'num_radar_pts'][i]
                empty_instance['bbox_3d_isvalid'] = ori_info_dict[
                    'valid_flag'][i]
                empty_instance = clear_instance_unused_keys(empty_instance)
                temp_data_info['instances'].append(empty_instance)
            temp_data_info[
                'cam_instances'] = generate_nuscenes_camera_instances(
                    ori_info_dict, nusc)
        if 'pts_semantic_mask_path' in ori_info_dict:
            temp_data_info['pts_semantic_mask_path'] = Path(
                ori_info_dict['pts_semantic_mask_path']).name
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'nuscenes'
    metainfo['version'] = data_list['metadata']['version']
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def generate_nuscenes_camera_instances(info, nusc):

    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

    empty_multicamera_instance = get_empty_multicamera_instances(camera_types)

    for cam in camera_types:
        cam_info = info['cams'][cam]
        # list[dict]
        ann_infos = get_nuscenes_2d_boxes(
            nusc,
            cam_info['sample_data_token'],
            visibilities=['', '1', '2', '3', '4'])
        empty_multicamera_instance[cam] = ann_infos

    return empty_multicamera_instance


def update_s3dis_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {'classes': ('table', 'chair', 'sofa', 'bookcase', 'board')}
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(mmengine.track_iter_progress(data_list)):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['sample_idx'] = i
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['pts_path']).name
        if 'pts_semantic_mask_path' in ori_info_dict:
            temp_data_info['pts_semantic_mask_path'] = Path(
                ori_info_dict['pts_semantic_mask_path']).name
        if 'pts_instance_mask_path' in ori_info_dict:
            temp_data_info['pts_instance_mask_path'] = Path(
                ori_info_dict['pts_instance_mask_path']).name

        # TODO support camera
        # np.linalg.inv(info['axis_align_matrix'] @ extrinsic): depth2cam
        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['class'])
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()

                    if anns['class'][instance_id] < len(METAINFO['classes']):
                        empty_instance['bbox_label_3d'] = anns['class'][
                            instance_id]
                    else:
                        ignore_class_name.add(
                            METAINFO['classes'][anns['class'][instance_id]])
                        empty_instance['bbox_label_3d'] = -1

                    empty_instance = clear_instance_unused_keys(empty_instance)
                    instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 's3dis'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_scannet_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['pts_path']).name
        if 'pts_semantic_mask_path' in ori_info_dict:
            temp_data_info['pts_semantic_mask_path'] = Path(
                ori_info_dict['pts_semantic_mask_path']).name
        if 'pts_instance_mask_path' in ori_info_dict:
            temp_data_info['pts_instance_mask_path'] = Path(
                ori_info_dict['pts_instance_mask_path']).name

        # TODO support camera
        # np.linalg.inv(info['axis_align_matrix'] @ extrinsic): depth2cam
        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            temp_data_info['axis_align_matrix'] = anns[
                'axis_align_matrix'].tolist()
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['name'])
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()

                    if anns['name'][instance_id] in METAINFO['classes']:
                        empty_instance['bbox_label_3d'] = METAINFO[
                            'classes'].index(anns['name'][instance_id])
                    else:
                        ignore_class_name.add(anns['name'][instance_id])
                        empty_instance['bbox_label_3d'] = -1

                    empty_instance = clear_instance_unused_keys(empty_instance)
                    instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'scannet'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_sunrgbd_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes': ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                    'dresser', 'night_stand', 'bookshelf', 'bathtub')
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['pts_path']).name
        calib = ori_info_dict['calib']
        rt_mat = calib['Rt']
        # follow Coord3DMode.convert_point
        rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                           ]) @ rt_mat.transpose(1, 0)
        depth2img = calib['K'] @ rt_mat
        temp_data_info['images']['CAM0']['depth2img'] = depth2img.tolist()
        temp_data_info['images']['CAM0']['img_path'] = Path(
            ori_info_dict['image']['image_path']).name
        h, w = ori_info_dict['image']['image_shape']
        temp_data_info['images']['CAM0']['height'] = h
        temp_data_info['images']['CAM0']['width'] = w

        anns = ori_info_dict.get('annos', None)
        if anns is not None:
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['name'])
                ignore_class_name = set()
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()
                    empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                    if anns['name'][instance_id] in METAINFO['classes']:
                        empty_instance['bbox_label_3d'] = METAINFO[
                            'classes'].index(anns['name'][instance_id])
                        empty_instance['bbox_label'] = empty_instance[
                            'bbox_label_3d']
                    else:
                        ignore_class_name.add(anns['name'][instance_id])
                        empty_instance['bbox_label_3d'] = -1
                        empty_instance['bbox_label'] = -1
                    empty_instance = clear_instance_unused_keys(empty_instance)
                    instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'sunrgbd'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_lyft_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    METAINFO = {
        'classes':
        ('car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
         'motorcycle', 'bicycle', 'pedestrian', 'animal'),
    }
    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(
            mmengine.track_iter_progress(data_list['infos'])):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['sample_idx'] = i
        temp_data_info['token'] = ori_info_dict['token']
        temp_data_info['ego2global'] = convert_quaternion_to_matrix(
            ori_info_dict['ego2global_rotation'],
            ori_info_dict['ego2global_translation'])
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict.get(
            'num_features', 5)
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['lidar_path']).name
        temp_data_info['lidar_points'][
            'lidar2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['lidar2ego_rotation'],
                ori_info_dict['lidar2ego_translation'])
        # bc-breaking: Timestamp has divided 1e6 in pkl infos.
        temp_data_info['timestamp'] = ori_info_dict['timestamp'] / 1e6
        for ori_sweep in ori_info_dict['sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            temp_lidar_sweep['lidar_points'][
                'lidar2ego'] = convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation'])
            temp_lidar_sweep['ego2global'] = convert_quaternion_to_matrix(
                ori_sweep['ego2global_rotation'],
                ori_sweep['ego2global_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_sweep['sensor2lidar_rotation']
            trans = ori_sweep['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            temp_lidar_sweep['lidar_points'][
                'lidar2sensor'] = lidar2sensor.astype(np.float32).tolist()
            # bc-breaking: Timestamp has divided 1e6 in pkl infos.
            temp_lidar_sweep['timestamp'] = ori_sweep['timestamp'] / 1e6
            temp_lidar_sweep['lidar_points']['lidar_path'] = ori_sweep[
                'data_path']
            temp_lidar_sweep['sample_data_token'] = ori_sweep[
                'sample_data_token']
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)
        temp_data_info['images'] = {}
        for cam in ori_info_dict['cams']:
            empty_img_info = get_empty_img_info()
            empty_img_info['img_path'] = Path(
                ori_info_dict['cams'][cam]['data_path']).name
            empty_img_info['cam2img'] = ori_info_dict['cams'][cam][
                'cam_intrinsic'].tolist()
            empty_img_info['sample_data_token'] = ori_info_dict['cams'][cam][
                'sample_data_token']
            empty_img_info[
                'timestamp'] = ori_info_dict['cams'][cam]['timestamp'] / 1e6
            empty_img_info['cam2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['cams'][cam]['sensor2ego_rotation'],
                ori_info_dict['cams'][cam]['sensor2ego_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_info_dict['cams'][cam]['sensor2lidar_rotation']
            trans = ori_info_dict['cams'][cam]['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            empty_img_info['lidar2cam'] = lidar2sensor.astype(
                np.float32).tolist()
            temp_data_info['images'][cam] = empty_img_info
        ignore_class_name = set()
        if 'gt_boxes' in ori_info_dict:
            num_instances = ori_info_dict['gt_boxes'].shape[0]
            for i in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox_3d'] = ori_info_dict['gt_boxes'][
                    i, :].tolist()
                if ori_info_dict['gt_names'][i] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        ori_info_dict['gt_names'][i])
                else:
                    ignore_class_name.add(ori_info_dict['gt_names'][i])
                    empty_instance['bbox_label'] = -1
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance = clear_instance_unused_keys(empty_instance)
                temp_data_info['instances'].append(empty_instance)
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'lyft'
    metainfo['version'] = data_list['metadata']['version']
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_waymo_infos(pkl_path, out_dir):
    # the input pkl is based on the
    # pkl generated in the waymo cam only challenage.
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_SIDE_LEFT',
        'CAM_SIDE_RIGHT',
    ]
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    # TODO update to full label
    # TODO discuss how to process 'Van', 'DontCare'
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist', 'Sign'),
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info(camera_types)

        if 'plane' in ori_info_dict:
            temp_data_info['plane'] = ori_info_dict['plane']
        temp_data_info['sample_idx'] = ori_info_dict['image']['image_idx']

        # calib matrix
        for cam_idx, cam_key in enumerate(camera_types):
            temp_data_info['images'][cam_key]['cam2img'] =\
                 ori_info_dict['calib'][f'P{cam_idx}'].tolist()

        for cam_idx, cam_key in enumerate(camera_types):
            rect = ori_info_dict['calib']['R0_rect'].astype(np.float32)
            velo_to_cam = 'Tr_velo_to_cam'
            if cam_idx != 0:
                velo_to_cam += str(cam_idx)
            Trv2c = ori_info_dict['calib'][velo_to_cam].astype(np.float32)

            lidar2cam = rect @ Trv2c
            temp_data_info['images'][cam_key]['lidar2cam'] = lidar2cam.tolist()
            temp_data_info['images'][cam_key]['lidar2img'] = (
                ori_info_dict['calib'][f'P{cam_idx}'] @ lidar2cam).tolist()

        # image path
        base_img_path = Path(ori_info_dict['image']['image_path']).name

        for cam_idx, cam_key in enumerate(camera_types):
            temp_data_info['images'][cam_key]['timestamp'] = ori_info_dict[
                'timestamp']
            temp_data_info['images'][cam_key]['img_path'] = base_img_path

        h, w = ori_info_dict['image']['image_shape']

        # for potential usage
        temp_data_info['images'][camera_types[0]]['height'] = h
        temp_data_info['images'][camera_types[0]]['width'] = w
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['timestamp'] = ori_info_dict[
            'timestamp']
        velo_path = ori_info_dict['point_cloud'].get('velodyne_path')
        if velo_path is not None:
            temp_data_info['lidar_points']['lidar_path'] = Path(velo_path).name

        # TODO discuss the usage of Tr_velo_to_cam in lidar
        Trv2c = ori_info_dict['calib']['Tr_velo_to_cam'].astype(np.float32)

        temp_data_info['lidar_points']['Tr_velo_to_cam'] = Trv2c.tolist()

        # for potential usage
        # temp_data_info['images']['R0_rect'] = ori_info_dict['calib'][
        #     'R0_rect'].astype(np.float32).tolist()

        # for the sweeps part:
        temp_data_info['timestamp'] = ori_info_dict['timestamp']
        temp_data_info['ego2global'] = ori_info_dict['pose']

        for ori_sweep in ori_info_dict['sweeps']:
            # lidar sweeps
            lidar_sweep = get_single_lidar_sweep()
            lidar_sweep['ego2global'] = ori_sweep['pose']
            lidar_sweep['timestamp'] = ori_sweep['timestamp']
            lidar_sweep['lidar_points']['lidar_path'] = Path(
                ori_sweep['velodyne_path']).name
            # image sweeps
            image_sweep = get_single_image_sweep(camera_types)
            image_sweep['ego2global'] = ori_sweep['pose']
            image_sweep['timestamp'] = ori_sweep['timestamp']
            img_path = Path(ori_sweep['image_path']).name
            for cam_idx, cam_key in enumerate(camera_types):
                image_sweep['images'][cam_key]['img_path'] = img_path

            temp_data_info['lidar_sweeps'].append(lidar_sweep)
            temp_data_info['image_sweeps'].append(image_sweep)

        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            num_instances = len(anns['name'])

            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        anns['name'][instance_id])
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    empty_instance['bbox_label'] = -1

                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                loc = anns['location'][instance_id]
                dims = anns['dimensions'][instance_id]
                rots = anns['rotation_y'][:, None][instance_id]
                gt_bboxes_3d = np.concatenate([loc, dims, rots
                                               ]).astype(np.float32).tolist()
                empty_instance['bbox_3d'] = gt_bboxes_3d
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                empty_instance['truncated'] = int(
                    anns['truncated'][instance_id].tolist())
                empty_instance['occluded'] = anns['occluded'][
                    instance_id].tolist()
                empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
                empty_instance['index'] = anns['index'][instance_id].tolist()
                empty_instance['group_id'] = anns['group_ids'][
                    instance_id].tolist()
                empty_instance['difficulty'] = anns['difficulty'][
                    instance_id].tolist()
                empty_instance['num_lidar_pts'] = anns['num_points_in_gt'][
                    instance_id].tolist()
                empty_instance['camera_id'] = anns['camera_id'][
                    instance_id].tolist()
                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list

        # waymo provide the labels that sync with cam
        anns = ori_info_dict.get('cam_sync_annos', None)
        ignore_class_name = set()
        if anns is not None:
            num_instances = len(anns['name'])
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        anns['name'][instance_id])
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    empty_instance['bbox_label'] = -1

                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                loc = anns['location'][instance_id]
                dims = anns['dimensions'][instance_id]
                rots = anns['rotation_y'][:, None][instance_id]
                gt_bboxes_3d = np.concatenate([loc, dims, rots
                                               ]).astype(np.float32).tolist()
                empty_instance['bbox_3d'] = gt_bboxes_3d
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                empty_instance['truncated'] = int(
                    anns['truncated'][instance_id].tolist())
                empty_instance['occluded'] = anns['occluded'][
                    instance_id].tolist()
                empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
                empty_instance['index'] = anns['index'][instance_id].tolist()
                empty_instance['group_id'] = anns['group_ids'][
                    instance_id].tolist()
                empty_instance['camera_id'] = anns['camera_id'][
                    instance_id].tolist()
                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            temp_data_info['cam_sync_instances'] = instance_list

            cam_instances = generate_waymo_camera_instances(
                ori_info_dict, camera_types)
            temp_data_info['cam_instances'] = cam_instances

        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'waymo'
    metainfo['version'] = '1.4'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def generate_waymo_camera_instances(ori_info_dict, cam_keys):

    empty_multicamera_instances = get_empty_multicamera_instances(cam_keys)

    for cam_idx, cam_key in enumerate(cam_keys):
        annos = copy.deepcopy(ori_info_dict['cam_sync_annos'])
        if cam_idx != 0:
            annos = convert_annos(ori_info_dict, cam_idx)

        ann_infos = get_kitti_style_2d_boxes(
            ori_info_dict, cam_idx, occluded=[0], annos=annos, dataset='waymo')

        empty_multicamera_instances[cam_key] = ann_infos
    return empty_multicamera_instances


def parse_args():
    parser = argparse.ArgumentParser(description='Arg parser for data coords '
                                     'update due to coords sys refactor.')
    parser.add_argument(
        '--dataset', type=str, default='kitti', help='name of dataset')
    parser.add_argument(
        '--pkl-path',
        type=str,
        default='./data/kitti/kitti_infos_train.pkl ',
        help='specify the root dir of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='converted_annotations',
        required=False,
        help='output direction of info pkl')
    args = parser.parse_args()
    return args



# Already present function: just takes the "dataset" argument and calls the right function
# Peculiarities:
#   - Adds two cases: 
#       - one for custom with images
#       - one for custom with NO images
#   - Does not use the lowercase because I want to use it (last two "elif" cases)
def update_pkl_infos(dataset, out_dir, pkl_path):
    if dataset.lower() == 'kitti':
        update_kitti_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'waymo':
        update_waymo_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'scannet':
        update_scannet_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'sunrgbd':
        update_sunrgbd_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'lyft':
        update_lyft_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'nuscenes':
        update_nuscenes_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 's3dis':
        update_s3dis_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset == 'michele_custom_images':
        update_michele_custom_infos(pkl_path, out_dir)
    elif dataset == 'michele_custom_NO_IMAGES':
        update_michele_custom_infos(pkl_path, out_dir, use_images=False)
    else:
        raise NotImplementedError(f'Do not support convert {dataset} to v2.')


if __name__ == '__main__':
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.root_dir
    update_pkl_infos(
        dataset=args.dataset, out_dir=args.out_dir, pkl_path=args.pkl_path)



"""
Printing "empty standard data" as output of "get_empty_standard_data_info()" function:
{'sample_idx': None, 'token': None, 'timestamp': None, 'ego2global': None, 'images': {'CAM0': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'CAM1': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'CAM2': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'CAM3': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'CAM4': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}}, 'lidar_points': {'num_pts_feats': None, 'lidar_path': None, 'lidar2ego': None}, 'radar_points': {'num_pts_feats': None, 'radar_path': None, 'radar2ego': None}, 'image_sweeps': [], 'lidar_sweeps': [], 'instances': [], 'instances_ignore': [], 'pts_semantic_mask_path': None, 'pts_instance_mask_path': None}



Instance 0: 
    "Original" format: 
    {'image': {'image_idx': 0, 'image_path': 'training/image_2/000000.png', 'image_shape': array([ 370, 1224], dtype=int32)}, 'point_cloud': {'num_features': 4, 'velodyne_path': 'training/velodyne/000000.bin'}, 'calib': {'P0': array([[707.0493,   0.    , 604.0814,   0.    ],
       [  0.    , 707.0493, 180.5066,   0.    ],
       [  0.    ,   0.    ,   1.    ,   0.    ],
       [  0.    ,   0.    ,   0.    ,   1.    ]]), 'P1': array([[ 707.0493,    0.    ,  604.0814, -379.7842],
       [   0.    ,  707.0493,  180.5066,    0.    ],
       [   0.    ,    0.    ,    1.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    1.    ]]), 'P2': array([[ 7.070493e+02,  0.000000e+00,  6.040814e+02,  4.575831e+01],
       [ 0.000000e+00,  7.070493e+02,  1.805066e+02, -3.454157e-01],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  4.981016e-03],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'P3': array([[ 7.070493e+02,  0.000000e+00,  6.040814e+02, -3.341081e+02],
       [ 0.000000e+00,  7.070493e+02,  1.805066e+02,  2.330660e+00],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  3.201153e-03],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'R0_rect': array([[ 0.9999128 ,  0.01009263, -0.00851193,  0.        ],
       [-0.01012729,  0.9999406 , -0.00403767,  0.        ],
       [ 0.00847068,  0.00412352,  0.9999556 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), 'Tr_velo_to_cam': array([[ 0.00692796, -0.9999722 , -0.00275783, -0.02457729],
       [-0.00116298,  0.00274984, -0.9999955 , -0.06127237],
       [ 0.9999753 ,  0.00693114, -0.0011439 , -0.3321029 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), 'Tr_imu_to_velo': array([[ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
       [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
       [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])}, 'annos': {'name': array(['Pedestrian'], dtype='<U10'), 'truncated': array([0.]), 'occluded': array([0]), 'alpha': array([-0.2]), 'bbox': array([[712.4 , 143.  , 810.73, 307.92]]), 'dimensions': array([[1.2 , 1.89, 0.48]]), 'location': array([[1.84, 1.47, 8.41]]), 'rotation_y': array([0.01]), 'score': array([0.]), 'index': array([0], dtype=int32), 'group_ids': array([0], dtype=int32), 'difficulty': array([0], dtype=int32), 'num_points_in_gt': array([377], dtype=int32)}}
    New "MMDet" format: 
    {'sample_idx': 0, 'token': None, 'timestamp': None, 'ego2global': None, 'images': {'CAM0': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[707.0493, 0.0, 604.0814, 0.0], [0.0, 707.0493, 180.5066, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -216.70103429706526], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.22322079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM1': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[707.0493, 0.0, 604.0814, -379.7842], [0.0, 707.0493, 180.5066, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -596.4852342970653], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.22322079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM2': {'img_path': '000000.png', 'height': 370, 'width': 1224, 'depth_map': None, 'cam2img': [[707.0493, 0.0, 604.0814, 45.75831], [0.0, 707.0493, 180.5066, -0.3454157], [0.0, 0.0, 1.0, 0.004981016], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -170.94272429706527], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.56863649623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.3275679897468414], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None, 'lidar2cam': [[-0.0015960992313921452, -0.9999162554740906, -0.012840436771512032, -0.022366708144545555], [-0.00527064548805356, 0.012848696671426296, -0.9999035596847534, -0.05967890843749046], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[707.0493, 0.0, 604.0814, -334.1081], [0.0, 707.0493, 180.5066, 2.33066], [0.0, 0.0, 1.0, 0.003201153], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -550.8091342970653], [176.77725024402156, 8.80879965630006, -707.9361204188122, -99.89256079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.3293478527468414], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM4': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'R0_rect': [[0.9999127984046936, 0.010092630051076412, -0.008511931635439396, 0.0], [-0.010127290152013302, 0.9999405741691589, -0.004037670791149139, 0.0], [0.008470674976706505, 0.0041235219687223434, 0.9999555945396423, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': '000000.bin', 'lidar2ego': None, 'Tr_velo_to_cam': [[0.006927963811904192, -0.9999722242355347, -0.0027578289154917, -0.024577289819717407], [-0.0011629819637164474, 0.0027498360723257065, -0.9999955296516418, -0.06127237156033516], [0.999975323677063, 0.006931141018867493, -0.0011438990477472544, -0.33210289478302], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'radar_points': {'num_pts_feats': None, 'radar_path': None, 'radar2ego': None}, 'image_sweeps': [], 'lidar_sweeps': [], 'instances': [{'bbox': [712.4, 143.0, 810.73, 307.92], 'bbox_label': 0, 'bbox_3d': [1.84, 1.47, 8.41, 1.2, 1.89, 0.48, 0.01], 'bbox_label_3d': 0, 'depth': 8.4149808883667, 'center_2d': [763.7633056640625, 224.4706268310547], 'num_lidar_pts': 377, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -0.2, 'score': 0.0, 'index': 0, 'group_id': 0}], 'instances_ignore': [], 'pts_semantic_mask_path': None, 'pts_instance_mask_path': None, 'cam_instances': {'CAM2': [{'bbox_label': 0, 'bbox_label_3d': 0, 'bbox': [710.4446301035068, 144.00207112943306, 820.2930685018162, 307.58688675239017], 'bbox_3d_isvalid': True, 'bbox_3d': [1.840000033378601, 1.4700000286102295, 8.40999984741211, 1.2000000476837158, 1.8899999856948853, 0.47999998927116394, 0.009999999776482582], 'velocity': -1, 'center_2d': [763.7633056640625, 224.4706268310547], 'depth': 8.4149808883667}]}}



Instance 1:
    "Original" format: 
    {'image': {'image_idx': 3, 'image_path': 'training/image_2/000003.png', 'image_shape': array([ 375, 1242], dtype=int32)}, 'point_cloud': {'num_features': 4, 'velodyne_path': 'training/velodyne/000003.bin'}, 'calib': {'P0': array([[721.5377,   0.    , 609.5593,   0.    ],
       [  0.    , 721.5377, 172.854 ,   0.    ],
       [  0.    ,   0.    ,   1.    ,   0.    ],
       [  0.    ,   0.    ,   0.    ,   1.    ]]), 'P1': array([[ 721.5377,    0.    ,  609.5593, -387.5744],
       [   0.    ,  721.5377,  172.854 ,    0.    ],
       [   0.    ,    0.    ,    1.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    1.    ]]), 'P2': array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
       [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
       [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
       [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]), 'P3': array([[ 7.215377e+02,  0.000000e+00,  6.095593e+02, -3.395242e+02],
       [ 0.000000e+00,  7.215377e+02,  1.728540e+02,  2.199936e+00],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  2.729905e-03],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'R0_rect': array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],
       [-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],
       [ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), 'Tr_velo_to_cam': array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
       [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
       [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'Tr_imu_to_velo': array([[ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
       [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
       [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])}, 'annos': {'name': array(['Car', 'DontCare', 'DontCare'], dtype='<U8'), 'truncated': array([ 0., -1., -1.]), 'occluded': array([ 0, -1, -1]), 'alpha': array([  1.55, -10.  , -10.  ]), 'bbox': array([[614.24, 181.78, 727.31, 284.77],
       [  5.  , 229.89, 214.12, 367.61],
       [522.25, 202.35, 547.77, 219.71]]), 'dimensions': array([[ 4.15,  1.57,  1.73],
       [-1.  , -1.  , -1.  ],
       [-1.  , -1.  , -1.  ]]), 'location': array([[    1.  ,     1.75,    13.22],
       [-1000.  , -1000.  , -1000.  ],
       [-1000.  , -1000.  , -1000.  ]]), 'rotation_y': array([  1.62, -10.  , -10.  ]), 'score': array([0., 0., 0.]), 'index': array([ 0, -1, -1], dtype=int32), 'group_ids': array([0, 1, 2], dtype=int32), 'difficulty': array([ 0,  0, -1], dtype=int32), 'num_points_in_gt': array([674,  -1,  -1], dtype=int32)}}
    New "MMDet" format:
    {'sample_idx': 3, 'token': None, 'timestamp': None, 'ego2global': None, 'images': {'CAM0': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, 0.0], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -167.8990963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM1': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, -387.5744], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -555.4734963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM2': {'img_path': '000003.png', 'height': 375, 'width': 1242, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -123.04181637996919], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.01668911726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2693869299305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None, 'lidar2cam': [[0.00023477392096538097, -0.9999441504478455, -0.01056347694247961, -0.002796816872432828], [0.010449407622218132, 0.010565354488790035, -0.999889612197876, -0.07510878890752792], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, -339.5242], [0.0, 721.5377, 172.854, 2.199936], [0.0, 0.0, 1.0, 0.002729905], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -507.4232963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -99.03313221726785], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2694029089305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM4': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'R0_rect': [[0.9999238848686218, 0.009837759658694267, -0.007445048075169325, 0.0], [-0.00986979529261589, 0.9999421238899231, -0.004278459120541811, 0.0], [0.007402527146041393, 0.0043516140431165695, 0.999963104724884, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': '000003.bin', 'lidar2ego': None, 'Tr_velo_to_cam': [[0.0075337449088692665, -0.9999713897705078, -0.00061660201754421, -0.004069766029715538], [0.01480249036103487, 0.0007280732970684767, -0.9998902082443237, -0.07631617784500122], [0.9998620748519897, 0.007523790001869202, 0.014807550236582756, -0.2717806100845337], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'radar_points': {'num_pts_feats': None, 'radar_path': None, 'radar2ego': None}, 'image_sweeps': [], 'lidar_sweeps': [], 'instances': [{'bbox': [614.24, 181.78, 727.31, 284.77], 'bbox_label': 2, 'bbox_3d': [1.0, 1.75, 13.22, 4.15, 1.57, 1.73, 1.62], 'bbox_label_3d': 2, 'depth': 13.222745895385742, 'center_2d': [667.3931274414062, 225.4925079345703], 'num_lidar_pts': 674, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.55, 'score': 0.0, 'index': 0, 'group_id': 0}, {'bbox': [5.0, 229.89, 214.12, 367.61], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': 0, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 1}, {'bbox': [522.25, 202.35, 547.77, 219.71], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 2}], 'instances_ignore': [], 'pts_semantic_mask_path': None, 'pts_instance_mask_path': None, 'cam_instances': {'CAM2': [{'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [615.6086242338421, 181.30492941287687, 727.8967407063733, 286.50770323470067], 'bbox_3d_isvalid': True, 'bbox_3d': [1.0, 1.75, 13.220000267028809, 4.150000095367432, 1.5700000524520874, 1.7300000190734863, 1.6200000047683716], 'velocity': -1, 'center_2d': [667.3931274414062, 225.4925079345703], 'depth': 13.222745895385742}]}}



Instance 2:
    "Original" format: 
    {'image': {'image_idx': 7, 'image_path': 'training/image_2/000007.png', 'image_shape': array([ 375, 1242], dtype=int32)}, 'point_cloud': {'num_features': 4, 'velodyne_path': 'training/velodyne/000007.bin'}, 'calib': {'P0': array([[721.5377,   0.    , 609.5593,   0.    ],
       [  0.    , 721.5377, 172.854 ,   0.    ],
       [  0.    ,   0.    ,   1.    ,   0.    ],
       [  0.    ,   0.    ,   0.    ,   1.    ]]), 'P1': array([[ 721.5377,    0.    ,  609.5593, -387.5744],
       [   0.    ,  721.5377,  172.854 ,    0.    ],
       [   0.    ,    0.    ,    1.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    1.    ]]), 'P2': array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
       [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
       [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
       [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]), 'P3': array([[ 7.215377e+02,  0.000000e+00,  6.095593e+02, -3.395242e+02],
       [ 0.000000e+00,  7.215377e+02,  1.728540e+02,  2.199936e+00],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  2.729905e-03],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'R0_rect': array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],
       [-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],
       [ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), 'Tr_velo_to_cam': array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
       [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
       [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'Tr_imu_to_velo': array([[ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
       [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
       [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])}, 'annos': {'name': array(['Car', 'Car', 'Car', 'Cyclist', 'DontCare', 'DontCare'],
      dtype='<U8'), 'truncated': array([ 0.,  0.,  0.,  0., -1., -1.]), 'occluded': array([ 0,  0,  0,  0, -1, -1]), 'alpha': array([ -1.56,   1.71,   1.64,   1.89, -10.  , -10.  ]), 'bbox': array([[564.62, 174.59, 616.43, 224.74],
       [481.59, 180.09, 512.55, 202.42],
       [542.05, 175.55, 565.27, 193.79],
       [330.6 , 176.09, 355.61, 213.6 ],
       [753.33, 164.32, 798.  , 186.74],
       [738.5 , 171.32, 753.27, 184.42]]), 'dimensions': array([[ 3.2 ,  1.61,  1.66],
       [ 3.7 ,  1.4 ,  1.51],
       [ 4.05,  1.46,  1.66],
       [ 1.95,  1.72,  0.5 ],
       [-1.  , -1.  , -1.  ],
       [-1.  , -1.  , -1.  ]]), 'location': array([[-6.900e-01,  1.690e+00,  2.501e+01],
       [-7.430e+00,  1.880e+00,  4.755e+01],
       [-4.710e+00,  1.710e+00,  6.052e+01],
       [-1.263e+01,  1.880e+00,  3.409e+01],
       [-1.000e+03, -1.000e+03, -1.000e+03],
       [-1.000e+03, -1.000e+03, -1.000e+03]]), 'rotation_y': array([ -1.59,   1.55,   1.56,   1.54, -10.  , -10.  ]), 'score': array([0., 0., 0., 0., 0., 0.]), 'index': array([ 0,  1,  2,  3, -1, -1], dtype=int32), 'group_ids': array([0, 1, 2, 3, 4, 5], dtype=int32), 'difficulty': array([ 0, -1, -1,  1, -1, -1], dtype=int32), 'num_points_in_gt': array([182,  20,   5,  25,  -1,  -1], dtype=int32)}}
    New "MMDet" format:
    {'sample_idx': 7, 'token': None, 'timestamp': None, 'ego2global': None, 'images': {'CAM0': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, 0.0], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -167.8990963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM1': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, -387.5744], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -555.4734963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM2': {'img_path': '000007.png', 'height': 375, 'width': 1242, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -123.04181637996919], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.01668911726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2693869299305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None, 'lidar2cam': [[0.00023477392096538097, -0.9999441504478455, -0.01056347694247961, -0.002796816872432828], [0.010449407622218132, 0.010565354488790035, -0.999889612197876, -0.07510878890752792], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, -339.5242], [0.0, 721.5377, 172.854, 2.199936], [0.0, 0.0, 1.0, 0.002729905], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -507.4232963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -99.03313221726785], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2694029089305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM4': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'R0_rect': [[0.9999238848686218, 0.009837759658694267, -0.007445048075169325, 0.0], [-0.00986979529261589, 0.9999421238899231, -0.004278459120541811, 0.0], [0.007402527146041393, 0.0043516140431165695, 0.999963104724884, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': '000007.bin', 'lidar2ego': None, 'Tr_velo_to_cam': [[0.0075337449088692665, -0.9999713897705078, -0.00061660201754421, -0.004069766029715538], [0.01480249036103487, 0.0007280732970684767, -0.9998902082443237, -0.07631617784500122], [0.9998620748519897, 0.007523790001869202, 0.014807550236582756, -0.2717806100845337], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'radar_points': {'num_pts_feats': None, 'radar_path': None, 'radar2ego': None}, 'image_sweeps': [], 'lidar_sweeps': [], 'instances': [{'bbox': [564.62, 174.59, 616.43, 224.74], 'bbox_label': 2, 'bbox_3d': [-0.69, 1.69, 25.01, 3.2, 1.61, 1.66, -1.59], 'bbox_label_3d': 2, 'depth': 25.012746810913086, 'center_2d': [591.3814697265625, 198.3730926513672], 'num_lidar_pts': 182, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -1.56, 'score': 0.0, 'index': 0, 'group_id': 0}, {'bbox': [481.59, 180.09, 512.55, 202.42], 'bbox_label': 2, 'bbox_3d': [-7.43, 1.88, 47.55, 3.7, 1.4, 1.51, 1.55], 'bbox_label_3d': 2, 'depth': 47.5527458190918, 'center_2d': [497.7289123535156, 190.75320434570312], 'num_lidar_pts': 20, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.71, 'score': 0.0, 'index': 1, 'group_id': 1}, {'bbox': [542.05, 175.55, 565.27, 193.79], 'bbox_label': 2, 'bbox_3d': [-4.71, 1.71, 60.52, 4.05, 1.46, 1.66, 1.56], 'bbox_label_3d': 2, 'depth': 60.52274703979492, 'center_2d': [554.121337890625, 184.53306579589844], 'num_lidar_pts': 5, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.64, 'score': 0.0, 'index': 2, 'group_id': 2}, {'bbox': [330.6, 176.09, 355.61, 213.6], 'bbox_label': 1, 'bbox_3d': [-12.63, 1.88, 34.09, 1.95, 1.72, 0.5, 1.54], 'bbox_label_3d': 1, 'depth': 34.09274673461914, 'center_2d': [343.5250549316406, 194.4336700439453], 'num_lidar_pts': 25, 'difficulty': 1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.89, 'score': 0.0, 'index': 3, 'group_id': 3}, {'bbox': [753.33, 164.32, 798.0, 186.74], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 4}, {'bbox': [738.5, 171.32, 753.27, 184.42], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 5}], 'instances_ignore': [], 'pts_semantic_mask_path': None, 'pts_instance_mask_path': None, 'cam_instances': {'CAM2': [{'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [565.4822720402807, 175.01202566042497, 616.6555088322534, 224.96047091220345], 'bbox_3d_isvalid': True, 'bbox_3d': [-0.6899999976158142, 1.690000057220459, 25.010000228881836, 3.200000047683716, 1.6100000143051147, 1.659999966621399, -1.590000033378601], 'velocity': -1, 'center_2d': [591.3814697265625, 198.3730926513672], 'depth': 25.012746810913086}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [481.8496708488522, 179.85710612050596, 512.4094377621442, 202.53901525985071], 'bbox_3d_isvalid': True, 'bbox_3d': [-7.429999828338623, 1.8799999952316284, 47.54999923706055, 3.700000047683716, 1.399999976158142, 1.5099999904632568, 1.5499999523162842], 'velocity': -1, 'center_2d': [497.7289123535156, 190.75320434570312], 'depth': 47.5527458190918}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [542.2247151650495, 175.73341152322814, 565.2443490828854, 193.9446887816074], 'bbox_3d_isvalid': True, 'bbox_3d': [-4.710000038146973, 1.7100000381469727, 60.52000045776367, 4.050000190734863, 1.4600000381469727, 1.659999966621399, 1.559999942779541], 'velocity': -1, 'center_2d': [554.121337890625, 184.53306579589844], 'depth': 60.52274703979492}, {'bbox_label': 1, 'bbox_label_3d': 1, 'bbox': [330.84191493374504, 176.13804311926262, 355.4978537323491, 213.8147876869614], 'bbox_3d_isvalid': True, 'bbox_3d': [-12.630000114440918, 1.8799999952316284, 34.09000015258789, 1.9500000476837158, 1.7200000286102295, 0.5, 1.5399999618530273], 'velocity': -1, 'center_2d': [343.5250549316406, 194.4336700439453], 'depth': 34.09274673461914}]}}

    

Instance 3:
    "Original" format: 
    {'image': {'image_idx': 9, 'image_path': 'training/image_2/000009.png', 'image_shape': array([ 375, 1242], dtype=int32)}, 'point_cloud': {'num_features': 4, 'velodyne_path': 'training/velodyne/000009.bin'}, 'calib': {'P0': array([[721.5377,   0.    , 609.5593,   0.    ],
       [  0.    , 721.5377, 172.854 ,   0.    ],
       [  0.    ,   0.    ,   1.    ,   0.    ],
       [  0.    ,   0.    ,   0.    ,   1.    ]]), 'P1': array([[ 721.5377,    0.    ,  609.5593, -387.5744],
       [   0.    ,  721.5377,  172.854 ,    0.    ],
       [   0.    ,    0.    ,    1.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    1.    ]]), 'P2': array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
       [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
       [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
       [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]), 'P3': array([[ 7.215377e+02,  0.000000e+00,  6.095593e+02, -3.395242e+02],
       [ 0.000000e+00,  7.215377e+02,  1.728540e+02,  2.199936e+00],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  2.729905e-03],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'R0_rect': array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],
       [-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],
       [ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), 'Tr_velo_to_cam': array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
       [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
       [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]), 'Tr_imu_to_velo': array([[ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
       [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
       [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])}, 'annos': {'name': array(['Car', 'Car', 'Car', 'DontCare', 'DontCare'], dtype='<U8'), 'truncated': array([ 0.,  0.,  0., -1., -1.]), 'occluded': array([ 0,  2,  0, -1, -1]), 'alpha': array([ -1.5 ,   1.75,   1.78, -10.  , -10.  ]), 'bbox': array([[601.96, 177.01, 659.15, 229.51],
       [600.14, 177.09, 624.65, 193.31],
       [574.98, 178.64, 598.45, 194.01],
       [710.6 , 167.73, 736.68, 182.35],
       [758.52, 156.27, 782.52, 179.23]]), 'dimensions': array([[ 3.2 ,  1.61,  1.66],
       [ 3.66,  1.44,  1.61],
       [ 3.37,  1.41,  1.53],
       [-1.  , -1.  , -1.  ],
       [-1.  , -1.  , -1.  ]]), 'location': array([[ 7.000e-01,  1.760e+00,  2.388e+01],
       [ 2.400e-01,  1.840e+00,  6.637e+01],
       [-2.190e+00,  1.960e+00,  6.825e+01],
       [-1.000e+03, -1.000e+03, -1.000e+03],
       [-1.000e+03, -1.000e+03, -1.000e+03]]), 'rotation_y': array([ -1.48,   1.76,   1.75, -10.  , -10.  ]), 'score': array([0., 0., 0., 0., 0.]), 'index': array([ 0,  1,  2, -1, -1], dtype=int32), 'group_ids': array([0, 1, 2, 3, 4], dtype=int32), 'difficulty': array([ 0, -1, -1, -1, -1], dtype=int32), 'num_points_in_gt': array([215,   4,   1,  -1,  -1], dtype=int32)}}
    New "MMDet" format:
    {'sample_idx': 9, 'token': None, 'timestamp': None, 'ego2global': None, 'images': {'CAM0': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, 0.0], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -167.8990963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM1': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, -387.5744], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -555.4734963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM2': {'img_path': '000009.png', 'height': 375, 'width': 1242, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -123.04181637996919], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.01668911726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2693869299305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None, 'lidar2cam': [[0.00023477392096538097, -0.9999441504478455, -0.01056347694247961, -0.002796816872432828], [0.010449407622218132, 0.010565354488790035, -0.999889612197876, -0.07510878890752792], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': [[721.5377, 0.0, 609.5593, -339.5242], [0.0, 721.5377, 172.854, 2.199936], [0.0, 0.0, 1.0, 0.002729905], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -507.4232963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -99.03313221726785], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2694029089305115], [0.0, 0.0, 0.0, 1.0]], 'cam2ego': None}, 'CAM4': {'img_path': None, 'height': None, 'width': None, 'depth_map': None, 'cam2img': None, 'lidar2img': None, 'cam2ego': None}, 'R0_rect': [[0.9999238848686218, 0.009837759658694267, -0.007445048075169325, 0.0], [-0.00986979529261589, 0.9999421238899231, -0.004278459120541811, 0.0], [0.007402527146041393, 0.0043516140431165695, 0.999963104724884, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': '000009.bin', 'lidar2ego': None, 'Tr_velo_to_cam': [[0.0075337449088692665, -0.9999713897705078, -0.00061660201754421, -0.004069766029715538], [0.01480249036103487, 0.0007280732970684767, -0.9998902082443237, -0.07631617784500122], [0.9998620748519897, 0.007523790001869202, 0.014807550236582756, -0.2717806100845337], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'radar_points': {'num_pts_feats': None, 'radar_path': None, 'radar2ego': None}, 'image_sweeps': [], 'lidar_sweeps': [], 'instances': [{'bbox': [601.96, 177.01, 659.15, 229.51], 'bbox_label': 2, 'bbox_3d': [0.7, 1.76, 23.88, 3.2, 1.61, 1.66, -1.48], 'bbox_label_3d': 2, 'depth': 23.88274574279785, 'center_2d': [632.515625, 201.69532775878906], 'num_lidar_pts': 215, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -1.5, 'score': 0.0, 'index': 0, 'group_id': 0}, {'bbox': [600.14, 177.09, 624.65, 193.31], 'bbox_label': 2, 'bbox_3d': [0.24, 1.84, 66.37, 3.66, 1.44, 1.61, 1.76], 'bbox_label_3d': 2, 'depth': 66.37274932861328, 'center_2d': [612.8189086914062, 185.02561950683594], 'num_lidar_pts': 4, 'difficulty': -1, 'truncated': 0.0, 'occluded': 2, 'alpha': 1.75, 'score': 0.0, 'index': 1, 'group_id': 1}, {'bbox': [574.98, 178.64, 598.45, 194.01], 'bbox_label': 2, 'bbox_3d': [-2.19, 1.96, 68.25, 3.37, 1.41, 1.53, 1.75], 'bbox_label_3d': 2, 'depth': 68.25274658203125, 'center_2d': [587.040283203125, 186.11753845214844], 'num_lidar_pts': 1, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.78, 'score': 0.0, 'index': 2, 'group_id': 2}, {'bbox': [710.6, 167.73, 736.68, 182.35], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 3}, {'bbox': [758.52, 156.27, 782.52, 179.23], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 4}], 'instances_ignore': [], 'pts_semantic_mask_path': None, 'pts_instance_mask_path': None, 'cam_instances': {'CAM2': [{'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [602.7258972606046, 177.07969133764107, 658.744415921266, 230.00911017200906], 'bbox_3d_isvalid': True, 'bbox_3d': [0.699999988079071, 1.7599999904632568, 23.8799991607666, 3.200000047683716, 1.6100000143051147, 1.659999966621399, -1.4800000190734863], 'velocity': -1, 'center_2d': [632.515625, 201.69532775878906], 'depth': 23.88274574279785}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [600.2515788427056, 177.07458691153306, 624.7781193061829, 193.45767369827445], 'bbox_3d_isvalid': True, 'bbox_3d': [0.23999999463558197, 1.840000033378601, 66.37000274658203, 3.6600000858306885, 1.440000057220459, 1.6100000143051147, 1.7599999904632568], 'velocity': -1, 'center_2d': [612.8189086914062, 185.02561950683594], 'depth': 66.37274932861328}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [575.1400930333315, 178.51572448611833, 598.421445977937, 194.1298090642433], 'bbox_3d_isvalid': True, 'bbox_3d': [-2.190000057220459, 1.9600000381469727, 68.25, 3.369999885559082, 1.409999966621399, 1.5299999713897705, 1.75], 'velocity': -1, 'center_2d': [587.040283203125, 186.11753845214844], 'depth': 68.25274658203125}]}}
"""