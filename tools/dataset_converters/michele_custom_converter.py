# Same imports that were in "kitti_converter.py"
from collections import OrderedDict
from pathlib import Path

import mmcv
import mmengine
import numpy as np
from nuscenes.utils.geometry_utils import view_points

from mmdet3d.structures import points_cam2img
from mmdet3d.structures.ops import box_np_ops
from .michele_custom_data_utils import get_michele_custom_image_info
from .nuscenes_converter import post_process_coords



# Copied function: reads the indeces from the ".txt" files
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]



# Copied function: adds the number of LiDAR points for each instance in the "annos" field
# Peculiarities:
#   - Uses the boolean "use_images" that deactivates the creation of data for the images
def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                use_images,
                                remove_outside=True,
                                num_features=4):
    for info in mmengine.track_iter_progress(infos):
        # Initialize the processing of the point-cloud
        pc_info = info['point_cloud']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        # Get the pointers to the needed "calib" dictionaries
        calib = info['calib']
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        # If using images, resize the point-cloud
        if use_images:                                                          ## Used the "use_images" boolean here
            image_info = info['image']
            if remove_outside:
                points_v = box_np_ops.remove_outside_points(                        ## Uses a complex "box_np_ops" function
                    points_v, rect, Trv2c, P2, image_info['image_shape'])
            # points_v = points_v[points_v[:, 0] > 0]
        # Compute the number of "not-Don't-Care" objects
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
            # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        # Take the information of the ground truth bounding boxes (dimension, location, rotation)
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        # Create a "camera_ref" vector and use a function to convert it to "velodyne_ref"
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(                                ## Uses a complex "box_np_ops" function
            gt_boxes_camera, rect, Trv2c)
        # Count the number of LiDAR points in the point-cloud
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)           ## Uses a complex "box_np_ops" function
        num_points_in_gt = indices.sum(0)
        # Just add a "-1" for the instances of "Don't Care" class
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        # Finally add the new key to the "annotations" dictionary
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)



# Copied function: main function to create the info files
# Peculiarities:
#   - Does not consider "with_plane" because don't want to use it
#   - Ignores "save_path" because was not used
#   - Uses the boolean "use_images" that deactivates the creation of data for the images 
def create_michele_custom_info_file(data_path,
                                    pkl_prefix,
                                    use_images,
                                    relative_path=True):
    """
    TODO Add documentation about the function?
    """

    # Gather the indeces of the various datasets (the results are lists)
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    # Same print as KITTI dataset, for reference
    print('Generate info. this may take several minutes.')
    # Define the path for saving the files that will be created (--> same as the path for the data)
    save_path = Path(data_path)

    # Obtain the information for the ".pkl" file for the training
    michele_custom_infos_train = get_michele_custom_image_info(
        data_path,
        use_images,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    # Add the number of LiDAR points for each instance in the "annos" field
    _calculate_num_points_in_gt(data_path, michele_custom_infos_train, relative_path, use_images)
    # Save the dictionary on the related ".pkl" file
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Michele_custom info train file is saved to {filename}')
    mmengine.dump(michele_custom_infos_train, filename)

    # Do the same for VAL files
    michele_custom_infos_val = get_michele_custom_image_info(
        data_path,
        use_images,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, michele_custom_infos_val, relative_path, use_images)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Michele_custom info val file is saved to {filename}')
    mmengine.dump(michele_custom_infos_val, filename)

    # Do the same for TRAIN-VAL files                                   ## Here, dictionaries are already created
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Michele_custom info trainval file is saved to {filename}')
    mmengine.dump(michele_custom_infos_train + michele_custom_infos_val, filename)

    # Do the same for TEST files                                        ## Here, skip calculating the number of points (do not need gt data, apparently)
    michele_custom_infos_test = get_michele_custom_image_info(
        data_path,
        use_images,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Michele_custom info test file is saved to {filename}')
    mmengine.dump(michele_custom_infos_test, filename)



# Copied function: function to "cut" the point clouds depending on the FOV of the cameras 
# Peculiarities:
#   - ignores "with_back" because always set to "False"
def create_michele_custom_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str, optional): Path of training set info.
            Default: None.
        val_info_path (str, optional): Path of validation set info.
            Default: None.
        test_info_path (str, optional): Path of test set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
        with_back (bool, optional): Whether to flip the points to back.
            Default: False.
    """
    # Create the paths to the ".pkl" info files
    if train_info_path is None:
        train_info_path = Path(data_path) / f'{pkl_prefix}_infos_train.pkl'
    if val_info_path is None:
        val_info_path = Path(data_path) / f'{pkl_prefix}_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / f'{pkl_prefix}_infos_test.pkl'

    # Create the actual REDUCED point clouds one dataset per time
    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    print('create reduced point cloud for validation set')
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(data_path, test_info_path, save_path)



# Copied function: actual cutter of the point cloud in account of "create_reduced_point_cloud"
# Peculiarities:
#   - ignores "with_back" because always set to "False"
def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False,
                                num_features=4,
                                front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str, optional): Path to save reduced point cloud
            data. Default: None.
        back (bool, optional): Whether to flip the points to back.
            Default: False.
        num_features (int, optional): Number of point features. Default: 4.
        front_camera_id (int, optional): The referenced/front camera ID.
            Default: 2.
    """
    # Load the whole dictionary from the ".pkl" info file path
    kitti_infos = mmengine.load(info_path)

    # Process the the point clouds one-by-one
    for info in mmengine.track_iter_progress(kitti_infos):
        # Get the pointers to the main dictionaries of the specific instance
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        # Get the pointer to the "rect" variable for the transformation
        rect = calib['R0_rect']

        # Get the path to the specific point cloud
        v_path = pc_info['velodyne_path']
        v_path = Path(data_path) / v_path
        
        # Load the points on a vector --> Checked: the vector is a [n][4] array with n="number of LiDAR points"
        points_v = np.fromfile(
            str(v_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])
        # Select the right camera number, and the the camera projection matrix
        if front_camera_id == 2:
            P2 = calib['P2']
        else:
            P2 = calib[f'P{str(front_camera_id)}']
        # Get the trasformation from "cam_ref" to "velo_ref"
        Trv2c = calib['Tr_velo_to_cam']
        
        # Use a dedicated function to remove the points that are outside of the camera FOV
            # first remove z < 0 points
            # keep = points_v[:, -1] > 0
            # points_v = points_v[keep]
            # then remove outside.
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,                      ## Uses a complex "box_np_ops" function
                                                    image_info['image_shape'])
        
        # Eventually create a new directory for the point clouds, and create the name of the new p.c.-file
        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
        else:
            save_filename = str(Path(save_path) / v_path.name)
        
        # Save the "reduced" points back to the file
        with open(save_filename, 'w') as f:
            points_v.tofile(f)