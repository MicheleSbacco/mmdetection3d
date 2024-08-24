# Same imports that were in "kitti_data_utils.py"
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np
from PIL import Image
from skimage import io

import math



"""
Set of functions copied straight-away from the kitti file
"""
def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)

def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2',
                   file_tail='.png',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)

def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)

def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check, use_prefix_id)

def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)



######################################### ATTENTION ##########################################
# Function has been deeply modified to be compliant with the reduced number of information that is available for
# the POLIMOVE dataset.
# Also, the parameter "use_images" has been added because the field "bbox" (pixels of the image corresponding to
# the bbox corners) could be useful in case of image use.
#
# For information about the fields/what they mean/etc, see file "create_data.py" and search "More info about the data processing".

def get_label_anno(label_path, use_images=False):
    # Create an empty instance of the "annotations" dictionary
    annotations = {}
    annotations.update({
        'name': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    # Get a variable "content" which is a list of lists where each sub-list contains the data about one single instance
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    # Count the number of "relevant" objects and the number of "total" objects
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    num_gt = len(content)
    # Take the name as the first field of the ".txt" file
    annotations['name'] = np.array([x[0] for x in content])
    # dimensions will convert hwl format to standard lhw(camera) format.
    # As said by the comment above, they want a (length-height-width) format. So we need to make up for it
    # TODO: Comment on the final version
    # annotations['dimensions'] = np.array([[float(info) for info in x[4:7]]
    #                                       for x in content
    #                                       ]).reshape(-1, 3)[:, [0, 2, 1]]
    annotations['dimensions'] = np.array([[float(info) for info in x[4:7]]
                                          for x in content
                                          ]).reshape(-1, 3)
    # Take the annotations in the frame of POLIMOVE, and then update them to the frame of KittiCamera
    annotations['location'] = np.array([[float(info) for info in x[1:4]]
                                        for x in content]).reshape(-1, 3)
    # TODO: Add comment on the change
    # for i, position in enumerate(annotations['location']):
    #     new_z = position[0]
    #     new_y = -position[2]
    #     new_x = -position[1]
    #     annotations['location'][i][0] = new_x
    #     annotations['location'][i][1] = new_y
    #     annotations['location'][i][2] = new_z
    # Take the rotation in degrees (POLIMOVE) and turn it into radians (KittiCamera)    TODO: Check that the sign is right (one axis upward, other one downward)
    #                                                                                   TODO: If I do not manipulate the pointcloud, why should I change the bboxes?
    annotations['rotation_y'] = np.array([float(x[7])
                                          for x in content]).reshape(-1)
    # Create this two "strange" values TODO: Understand what are they needed for
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    # Just if images are used, add a field "bbox" to the annotations. For more info read the disclaimer before the
    # name of the function. 
    # TODO: assign the number of the fields according to what SUSTechPoints says and assigns
    if use_images:
        annotations.update({'bbox':[]})
        annotations['bbox'] = np.array([[float(info) for info in x[boh_riguarda1:boh_riguarda2]]
                                    for x in content]).reshape(-1, 4)
    return annotations



def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=bool)
    moderate_mask = np.ones((len(dims), ), dtype=bool)
    hard_mask = np.ones((len(dims), ), dtype=bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff
"""
End of the copied functions
"""


# Copied funcion: Creates the dictionary for the information in the "Kitti.pkl" format...
#                 ...THEN the dictionary will be updated to the "MMDet.pkl" format later on (after being saved)
# Peculiarities:
#   - Ignores "with_plane" because don't use it
#   - Uses the boolean "use_images" that deactivates the creation of data for the images 
def get_minerva_polimove_image_info(path,
                                  use_images,
                                  training=True,
                                  label_info=True,
                                  velodyne=False,
                                  calib=False,
                                  image_ids=7481,
                                  extend_matrix=True,
                                  num_worker=8,
                                  relative_path=True,
                                  with_imageshape=True):

    # Set the path and make sure that indeces are a list
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    # Define a function that will be iterated. Extracts data from the files and saves it into a dictionary
    def map_func(idx):
        
        # Instanciate dictionaries:     - main "info" dictionary, that will contain all the others
        #                               - "pc_info" for the point cloud
        #                               - ONLY IF NEEDED (differently from "michele_custom", here we directly remove calib and image, already at this stage):
        #                                   - "calib" for calibration
        #                                   - "image"
        #                               - "annotations" for the ground truth
        info = {}
        pc_info = {'num_features': 4}
        if use_images:                                                              ## Used the "use_images" boolean here
            calib_info = {}
            image_info = {'image_idx': idx}
        pc_info['pc_idx'] = idx                                                     ##  When working with only lidar, still need to store the index
        annotations = None
        
        # Update the "pc_info" dictionary with the path to the ".bin" file
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(idx, path, training, relative_path)
        
        # Update the "image" dictionary with the path and the image_shape
        if use_images:                                                                                  ## Used the "use_images" boolean here
            image_info['image_path'] = get_image_path(idx, path, training, relative_path)
            if with_imageshape:
                img_path = image_info['image_path']
                if relative_path:
                    img_path = str(root_path / img_path)
                image_info['image_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
        
        # Update the "annotations" dictionary
        if label_info:
            # Here, just update the path
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            # Here, add all the needed annotations with the specified function
            annotations = get_label_anno(label_path, use_images=use_images)
        
        # Add the "image","pc_info" dictionaries to the main "info" dictionary
        if use_images:                                                                          ## Used the "use_images" boolean here
            info['image'] = image_info
        info['point_cloud'] = pc_info
        
        
        # Only if needed, update the "calib" dictionary, and add it to the main "info" dictionary
        if use_images:                                                                          ## Used the "use_images" boolean here
            if calib:
                calib_path = get_calib_path(
                    idx, path, training, relative_path=False)
                with open(calib_path, 'r') as f:
                    lines = f.readlines()
                P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                            ]).reshape([3, 4])
                P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                            ]).reshape([3, 4])
                P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                            ]).reshape([3, 4])
                P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                            ]).reshape([3, 4])
                if extend_matrix:
                    P0 = _extend_matrix(P0)
                    P1 = _extend_matrix(P1)
                    P2 = _extend_matrix(P2)
                    P3 = _extend_matrix(P3)
                R0_rect = np.array([
                    float(info) for info in lines[4].split(' ')[1:10]
                ]).reshape([3, 3])
                if extend_matrix:
                    rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                    rect_4x4[3, 3] = 1.
                    rect_4x4[:3, :3] = R0_rect
                else:
                    rect_4x4 = R0_rect
                Tr_velo_to_cam = np.array([
                    float(info) for info in lines[5].split(' ')[1:13]
                ]).reshape([3, 4])
                Tr_imu_to_velo = np.array([
                    float(info) for info in lines[6].split(' ')[1:13]
                ]).reshape([3, 4])
                if extend_matrix:
                    Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                    Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
                calib_info['P0'] = P0
                calib_info['P1'] = P1
                calib_info['P2'] = P2
                calib_info['P3'] = P3
                calib_info['R0_rect'] = rect_4x4
                calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
                calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
                info['calib'] = calib_info
        # Add the "annotations" dictionary to the main "info" dictionary
        if annotations is not None:
            info['annos'] = annotations
            # Removing the line that added the difficulty here, by using the function "add_difficulty_to_annos"
            #add_difficulty_to_annos(info)
        # Return the main "info" dictionary
        return info

    # Iterate the "map" function and return the data in "raw" format
    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)