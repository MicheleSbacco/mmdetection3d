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

def get_label_anno(label_path, use_images):
    # Create an empty instance of the "annotations" dictionary
    annotations = {}
    annotations.update({
        'name': [],
        # 'truncated': [],  # Removed from the POLIMOVE dataset
        # 'occluded': [],   # Removed from the POLIMOVE dataset
        # 'alpha': [],      # Removed from the POLIMOVE dataset
        # 'bbox': [],           # Used only in case of images
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
    # Comments:
    #   - KITTI saves the dimensions in hwl format (height-width-length) while MMDetection3D-Camera saves it in lhw 
    #     format, so they need to re-order them from KITTI to MMDetection3D 
    #   - but we save them in lwh and then use the MMDetection3D-LiDAR format which is lwh, so no need to change
    annotations['dimensions'] = np.array([[float(info) for info in x[4:7]]
                                          for x in content]).reshape(-1, 3)
    # Take the annotations in the frame of POLIMOVE, and then update them to the frame of KittiCamera
    annotations['location'] = np.array([[float(info) for info in x[1:4]]
                                        for x in content]).reshape(-1, 3)
    # Add the rotation from the right field
    annotations['rotation_y'] = np.array([float(x[7])
                                          for x in content]).reshape(-1)
    # Create this two values:
    #   - First one is to give a number to the "important" instances (and "-1" to the DontCare objects)
    #   - Second one is to give a number to every instance
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    # Just if images are used, add a field "bbox" to the annotations (pixels of the image corresponding to
    # the corners of the bbox)
    if use_images:
        annotations.update({'bbox':[]})
        annotations['bbox'] = np.array([[float(info) for info in x[8:12]]
                                    for x in content]).reshape(-1, 4)
    return annotations



def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

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
                                  is_augmented,
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
        pc_info = {'num_features': 4} if not is_augmented else {'num_features': 5}
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
            annotations = get_label_anno(label_path, use_images)
        
        # Add the "image","pc_info" dictionaries to the main "info" dictionary
        if use_images:                                                                          ## Used the "use_images" boolean here
            info['image'] = image_info
        info['point_cloud'] = pc_info
        
        
        # Only if needed, update the "calib" dictionary, and add it to the main "info" dictionary
        #
        # NOTE: Highly modified to match with the POLIMOVE dataset
        #   - We don't have FOUR projection matrices for four different cameras, but just one for
        #     one camera
        #     ------> will call it P0
        #     ------> will be in line number 0
        #   - We don't use the R0_rect matrix since we don't use stereo cameras
        #   - We don't use the Tr_imu_to_velo
        #     ------> will be in line number 1
        if use_images:                                                                          ## Used the "use_images" boolean here
            if calib:
                calib_path = get_calib_path(
                    idx, path, training, relative_path=False)
                with open(calib_path, 'r') as f:
                    lines = f.readlines()
                P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                            ]).reshape([3, 4])
                if extend_matrix:
                    P0 = _extend_matrix(P0)
                Tr_velo_to_cam = np.array([
                    float(info) for info in lines[1].split(' ')[1:13]
                ]).reshape([3, 4])
                if extend_matrix:
                    Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                calib_info['P0'] = P0
                calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
                info['calib'] = calib_info
        # Add the "annotations" dictionary to the main "info" dictionary
        if annotations is not None:
            info['annos'] = annotations
        # Return the main "info" dictionary
        return info

    # Iterate the "map" function and return the data in "raw" format
    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)
