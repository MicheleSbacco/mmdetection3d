# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from mmengine import print_log

from tools.dataset_converters import indoor_converter as indoor
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters import lyft_converter as lyft_converter
from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from tools.dataset_converters import semantickitti_converter
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database, create_michele_custom_groundtruth_database, create_minerva_polimove_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos






'''
######################################### GENERAL DESCRIPTION OF THE DATA PROCESSING #########################################
'''

# What this whole file does: it parses the arguments and calls a function to create data just for
#       a specific dataset. In our case it calls the "kitti_data_prep" function with same "root"
#       and "out" directory, "info_prefix"="extra_tag", no "version" and "with_plane"=False.
#                                   |
#                                   |
#                                   |
#                                   V
# The "kitti_data_prep" function does a lot of stuff calling other complex functions:
#       - "create_kitti_info_file":
#               - gathers the image indeces from the ".txt" files
#               - generates "...infos....pkl" files for train,val,trainval,test with another 
#                 subfunction "get_kitti_image_info"
#               - the "_calculate_num_points_in_gt" apparently adds in the "annos" field the number 
#                 of LiDAR points for each instance
#               - the format of ".pkl" files is in tools/dataset_converters/kitti_data_utils.py
#                 in the function "get_kitti_image_info"
#       - "create_reduced_point_cloud":
#               - starting from ".pkl" files, this function creates folders with reduced point
#                 clouds 
#               - the "reduction" is based on the dimensions of the FOV of the camera
#       - "update_pkl_infos":
#               - just updates the ".pkl" files to the standard format in OpenMMLab V2.0
#       - "create_groundtruth_database":
#               - Creates the dictionary of a dataset and then does DATASETS.build()
#               - Has a "for" loop for each instance of the dataset, and another "for" loop for
#                 each bounding box.
#                 Then creates a file for each single bounding box and puts it into a folder in the
#                 form of ".bin" files
#               - Then finally puts all the info inside a ".pkl" file that is distinguished by the 
#                 "...db_infos....pkl" file name (others were just "...infos...")

# What we will have to do: create a function that responds to the command 
#       "python3 tools/create_data.py custom --root-path ./data/custom --out-dir ./data/custom --extra-tag custom"
#       and creates some ".pkl" files to be used for training

# More info about the data processing:
#       - from file "Notes for custom creation.txt" which has been copied at the end of this file as a comment
#       - in "update_infos_to_v2" there are some examples of data structures
#       - in "create_gt_database" there are some examples of data structures






'''
######################################### OLD PART ABOUT LIDAR_ONLY #########################################
'''

# Added imports
from dataset_converters import michele_custom_converter as mcc
# Added function that resembles "kitti_data_prep" but with LiDAR sensor only.
# Peculiarities:
#   - Does not implement the use of the ground, it directly removes it
#   - Adds a boolean "use_images" that determines if images are processed
#                           |
#                           |
#                           V
# TODO: Why not directly convert to the right format?
#
#
# STANDARDIZED COMMAND is: "python3 tools/create_data.py michele_custom --root-path ./data/michele_custom --out-dir ./data/michele_custom --extra-tag michele_custom --remove-images"   
def michele_custom_data_prep(root_path,
                             info_prefix,
                             version,
                             out_dir,
                             use_images):
    
    # Create the ".pkl" files
    mcc.create_michele_custom_info_file(root_path, info_prefix, use_images)
    
    # Create the "reduced" point clouds, only if use images
    if use_images:                                                                              ## Used the "use_images" boolean here
        mcc.create_michele_custom_reduced_point_cloud(root_path, info_prefix)

    # Based on the "use_images" boolean, give the dataset an appropriate name.
    # The function "update_pkl_infos" below updates the data from "kitti.pkl" format to "MMLab.pkl" format
    if use_images: dataset_for_update = 'michele_custom_images'                                         ## Used the "use_images" boolean here
    else: dataset_for_update = 'michele_custom_NO_IMAGES'
    # For the TRAINING:
    #   1. Create the path for the updated ".pkl" file (already existing but in "wrong format")
    #   2. Update the data using the dedicated function
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    update_pkl_infos(dataset_for_update, out_dir, info_train_path)                                  ## Uses a complex function in "update_infos_to_v2"
    # Same for VAL
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos(dataset_for_update, out_dir, info_val_path)                                    ## Same
    # Same for TRAIN-VAL
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    update_pkl_infos(dataset_for_update, out_dir=out_dir, pkl_path=info_trainval_path)              ## Same
    # Same for TEST
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos(dataset_for_update, out_dir=out_dir, pkl_path=info_test_path)                  ## Same

    # Function that creates the ground-truth ".pkl" file, and all the gt-files about each single instance
    #   - If use images, essentially the same as KittiDataset so we just use that one
    #   - If not using them, need to take the custom dataset
    if use_images: dataset_name="KittiDataset"
    else: dataset_name="MicheleCustomDatasetNoImages"
    create_michele_custom_groundtruth_database(
        dataset_name,
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl')






'''
######################################### MINERVA POLIMOVE ##########################################
'''


# Added import with the new data converter (specific for POLIMOVE data)
from dataset_converters import minerva_polimove_converter as mpc
# Just needed when debugging to stop the process
import time


# Added function that resembles "kitti_data_prep" but for Minerva custom data. 
# 
# TODO: For now only implemented for LiDAR, fill in for cameras when needed
# 
# Peculiarities:
#   - Does not implement the use of the ground, it directly removes it
#   - Adds a boolean "use_images" that determines if images are processed
#
# STANDARDIZED COMMAND is:  "python3 tools/create_data.py minerva_polimove --root-path ./data/michele_custom --out-dir ./data/michele_custom --extra-tag michele_custom --remove-images"
#                           "python3 tools/create_data.py   minerva_polimove                    (start the code pipeline)
#                                                           --root-path ./data/michele_custom   (where to find the files)
#                                                           --out-dir ./data/michele_custom     (where to put the data files, usually same as "root_path")
#                                                           --extra-tag michele_custom          (just gives the name to the data)
#                                                           --remove-images"                    (if inserted, images and calibration are ignored)
def minerva_polimove_data_prep(root_path,
                             info_prefix,
                             version,
                             out_dir,
                             use_images):
    
    # Create the ".pkl" files
    mpc.create_minerva_polimove_info_file(root_path, info_prefix, use_images)
    
    # Create the "reduced" point clouds, only if use images. TODO: Define appropriate function for images when needed
    if use_images:                                                                              ## Used the "use_images" boolean here
        mpc.create_minerva_polimove_reduced_point_cloud(root_path, info_prefix)

    # Based on the "use_images" boolean, give the dataset an appropriate name.
    # The function "update_pkl_infos" updates the data from "kitti.pkl" format to "MMLab.pkl" format
    if use_images: minerva_dataset_choice = 'minerva_polimove_cameralidar'                                     ## Used the "use_images" boolean here
    else: minerva_dataset_choice = 'minerva_polimove_lidaronly'
    # For the TRAINING:
    #   1. Create the path for the updated ".pkl" file (already existing but in "wrong format")
    #   2. Update the data using the dedicated function
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    update_pkl_infos(minerva_dataset_choice, out_dir, info_train_path)
    # Same for VAL
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos(minerva_dataset_choice, out_dir, info_val_path)
    # Same for TRAIN-VAL
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    update_pkl_infos(minerva_dataset_choice, out_dir=out_dir, pkl_path=info_trainval_path)
    # Same for TEST
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos(minerva_dataset_choice, out_dir=out_dir, pkl_path=info_test_path)
    
    # Function that creates the ground-truth ".pkl" file, and all the gt-files about each single instance
    #   - If use images, TODO complete here (for now completely empty)
    #   - If not using them, need to take the custom dataset
    create_minerva_polimove_groundtruth_database(
        minerva_dataset_choice,
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl')




def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    if version == 'v1.01-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_test_path)
    elif version == 'v1.01-train':
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_train_path)
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_val_path)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_test_path)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
    for split in splits:
        filename = osp.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
        update_pkl_infos('s3dis', out_dir=out_dir, pkl_path=filename)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_val_path)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=10,
                    only_gt_database=False,
                    save_senor_data=False,
                    skip_cam_instances_infos=False):
    """Prepare waymo dataset. There are 3 steps as follows:

    Step 1. Extract camera images and lidar point clouds from waymo raw
        data in '*.tfreord' and save as kitti format.
    Step 2. Generate waymo train/val/test infos and save as pickle file.
    Step 3. Generate waymo ground truth database (point clouds within
        each 3D bounding box) for data augmentation in training.
    Steps 1 and 2 will be done in Waymo2KITTI, and step 3 will be done in
    GTDatabaseCreater.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default to 10. Here we store ego2global information of these
            frames for later use.
        only_gt_database (bool, optional): Whether to only generate ground
            truth database. Default to False.
        save_senor_data (bool, optional): Whether to skip saving
            image and lidar. Default to False.
        skip_cam_instances_infos (bool, optional): Whether to skip
            gathering cam_instances infos in Step 2. Default to False.
    """
    from tools.dataset_converters import waymo_converter as waymo

    if version == 'v1.4':
        splits = [
            'training', 'validation', 'testing',
            'testing_3d_camera_only_detection'
        ]
    elif version == 'v1.4-mini':
        splits = ['training', 'validation']
    else:
        raise NotImplementedError(f'Unsupported Waymo version {version}!')
    out_dir = osp.join(out_dir, 'kitti_format')

    if not only_gt_database:
        for i, split in enumerate(splits):
            load_dir = osp.join(root_path, 'waymo_format', split)
            if split == 'validation':
                save_dir = osp.join(out_dir, 'training')
            else:
                save_dir = osp.join(out_dir, split)
            converter = waymo.Waymo2KITTI(
                load_dir,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split
                           in ['testing', 'testing_3d_camera_only_detection']),
                info_prefix=info_prefix,
                max_sweeps=max_sweeps,
                split=split,
                save_senor_data=save_senor_data,
                save_cam_instances=not skip_cam_instances_infos)
            converter.convert()
            if split == 'validation':
                converter.merge_trainval_infos()

        from tools.dataset_converters.waymo_converter import \
            create_ImageSets_img_ids
        create_ImageSets_img_ids(out_dir, splits)

    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()

    print_log('Successfully preparing Waymo Open Dataset')


def semantickitti_data_prep(info_prefix, out_dir):
    """Prepare the info file for SemanticKITTI dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    semantickitti_converter.create_semantickitti_info_file(
        info_prefix, out_dir)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='''Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!''')
parser.add_argument(
    '--skip-cam_instances-infos',
    action='store_true',
    help='''Whether to skip gathering cam_instances infos.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--skip-saving-sensor-data',
    action='store_true',
    help='''Whether to skip saving image and lidar.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--remove-images',
    action='store_false',
    default=True,
    help='''If written, disables images in the creation of ".pkl" files''')
args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    if args.dataset == 'kitti':
        if args.only_gt_database:
            create_groundtruth_database(
                'KittiDataset',
                args.root_path,
                args.extra_tag,
                f'{args.extra_tag}_infos_train.pkl',
                relative_path=False,
                mask_anno_path='instances_train.json',
                with_mask=(args.version == 'mask'))
        else:
            kitti_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,
                out_dir=args.out_dir,
                with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}-trainval'
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
            test_version = f'{args.version}-test'
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=test_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}'
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps,
            only_gt_database=args.only_gt_database,
            save_senor_data=not args.skip_saving_sensor_data,
            skip_cam_instances_infos=args.skip_cam_instances_infos)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'semantickitti':
        semantickitti_data_prep(
            info_prefix=args.extra_tag, out_dir=args.out_dir)
    elif args.dataset == 'michele_custom':
        michele_custom_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            use_images=args.remove_images)
    elif args.dataset == 'minerva_polimove':
        minerva_polimove_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            use_images=args.remove_images)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
    





'''
AT THE END OF FILE: Official documentation from KITTI Dataset






###########################################################################
#            HOW WE SAVE DATA AND FRAME ORIENTATION PROBLEM               #
###########################################################################

How we save data:
	- JSON
		{
		    "obj_id": "0",
		    "obj_type": "Car",
		    "psr": {
			"position": {
			    "x": 33.44053528495641,
			    "y": 13.468654831651865,
			    "z": 1.5964655633310705
			},
			"rotation": {
			    "x": 0,
			    "y": 0,
			    "z": -0.0037892504174532
			},
			"scale": {
			    "x": 5,
			    "y": 2,
			    "z": 1.5
			}
		    }
		}
	- TXT
		0    1     2     3     4      5     6      7      
		Name x_pos y_pos z_pos length width height rotation_vertical


Our axis are as follows:   - x: positive in front of the car
			   - y: positive to the left of the car
			   - z: positive going above the car
All information is given from the point of view of the COG (Centre Of Gravity) of the car.
Angles are expressed in degrees [deg]. Rotation is positive if CCW seen from above.


The axis for KITTI are instead: - x: points to the right of the camera
				- y: points downward (into the ground)
				- z: points forward in the direction the camera is facing
All the positions and locations in KITTI are expressed in camera coordinates.
Angles are expressed in radians [rad].
The 'DontCare' is used for objects that are "too far" in order to reduce the risk of false negatives






###########################################################################
#                 HOW MMDET SAVES DATA FROM KITTI                         #
###########################################################################

What we have for the use of KITTI with "create_data.py":
	- folder "ImageSets"
		- four folders with .txt files indicating which images/scans are for what (test, train, trainval, val)
	- folder "testing"
		- is for test, train, and trainval
		- has folders "calib", "Image_2" and "velodyne". Each of them has only raw data (no labels at all) in various formats (.txt, .png, .bin)
	- folder "training"
		- similar to folder "testing" but also has a folder "label_2"
		- in "label_2" there are .txt files containing the information of all the instances in each image/scan
		- each instance has a string and 14 numeric fields (15 in total), that are explained in (ASTHERISK_1)

Flow of information in the files:
	- FILE: "create_data.py"
		- FILE: "michele_custom_converter.py" from "kitti_converter.py"
			- FILE: "michele_custom_data_utils.py" from "kitti_data_utils.py"
				- FUNCT: "get_michele_custom_image_info". Different btw TRAIN/VAL and TEST.
					- Defines a function "map_func" which is then iterated on the list of image indeces
					- Creates a dictionary for each image/scan with this structure
						- point_cloud
							- num_features
							- pc_idx
							- velodyne_path
						- image
							- image_idx
							- image_path
							- image_shape
						- annos (not for TEST) (ASTHERISK_1)
							- name, @ index 0 (Pedestrian, Car, Misc, DontCare...)
							- truncated, @1
							- occluded, @2
							- alpha, @3
							- bbox, @4to7 (4 in total)
							- dimensions, @8to10 (3 in total)
							- location, @11to13 (3 in total)
							- rotation_y, @14
							- score, manually (the shape of the field "bbox")
							- index, manually (array from 0 to n-1 of the objects excluding the "DontCare", then as many -1's as there are DontCare's)
							- group_ids, manually (from 0 to k-1 including DontCare's)
							- difficulty, manually (strange with function "add_difficulty_to_annos")
						- calib (originally 3x4 matrices, if extend_matrix they all become 4x4 so homogeneous)
							- P0
							- P1
							- P2
							- P3
							- R0_rect (different treatment to maintain the data type, but still homogenous)
							- Tr_velo_to_cam
							- Tr_imu_to_velo
			- The result from previous is a list of dictionaries, one for each image/scan
			- FUNCT: "_calculate_num_points_in_gt"
				- adds a field "num_points_in_gt" to the "annos" field, with the number of points in the bboxes (excluded the DontCare's)
			- Finally dumps the info in a ".pkl" file
			- Repeats for train, val, trainval, test
		- FILE: same as above
			- done using the FUNCT: "create_michele_custom_reduced_point_cloud" 
				- Only if use_images
				- Cuts the point cloud of the .bin files according to what camera is used (usually number 2)
		- FILE: "update_infos_to_v2"
			- done using the FUNCT "update_pkl_infos" that sends to "update_michele_custom_infos"
			- at the end of the file, there is a comparison of the "original" and "MMDet3D" formats for some instances
			- There are two "for" loops
				- First one goes through all the instances of image/scan
				- Second one goes through all the instances in the "annos" field
			- Repeats for train, val, trainval, test
			- It looks like there is no change of coordinates in this part (TODO check, only remaining one is create_dataset.py)
		- FILE: "create_gt_database.py"
			- done using the FUNCT "create_michele_custom_groundtruth_database"
			- TODO: Finish this shit, and also understand where the fuck the transformation from camera to LiDAR happens






###########################################################################
#            THE KITTI VISION BENCHMARK SUITE: OBJECT BENCHMARK           #
#              Andreas Geiger    Philip Lenz    Raquel Urtasun            #
#                    Karlsruhe Institute of Technology                    #
#                Toyota Technological Institute at Chicago                #
#                             www.cvlibs.net                              #
###########################################################################

For recent updates see http://www.cvlibs.net/datasets/kitti/eval_object.php.

This file describes the KITTI 2D object detection and orientation estimation
benchmark, the 3D object detection benchmark and the bird's eye view benchmark.
The benchmarks consist of 7481 training images (and point clouds) 
and 7518 test images (and point clouds) for each task.
Despite the fact that we have labeled 8 different classes, only the
classes 'Car' and 'Pedestrian' are evaluated in our benchmark, as only for
those classes enough instances for a comprehensive evaluation have been
labeled. The labeling process has been performed in two steps: First we
hired a set of annotators, to label 3D bounding boxe tracklets in point
clouds. Since for a pedestrian tracklet, a single 3D bounding box tracklet
(dimensions have been fixed) often fits badly, we additionally labeled the
left/right boundaries of each object by making use of Mechanical Turk. We
also collected labels of the object's occlusion state, and computed the
object's truncation via backprojecting a car/pedestrian model into the
image plane.

NOTE: WHEN SUBMITTING RESULTS, PLEASE STORE THEM IN THE SAME DATA FORMAT IN
WHICH THE GROUND TRUTH DATA IS PROVIDED (SEE BELOW), USING THE FILE NAMES
000000.txt 000001.txt ... CREATE A ZIP ARCHIVE OF THEM AND STORE YOUR
RESULTS (ONLY THE RESULTS OF THE TEST SET) IN ITS ROOT FOLDER.

NOTE2: Please read the bottom of this file carefully if you plan to evaluate
results yourself on the training set.

NOTE3: WHEN SUBMITTING RESULTS FOR THE 3D OBJECT DETECTION BENCHMARK OR THE
BIRD'S EYE VIEW BENCHMARK (AS OF 2017), READ THE INSTRUCTIONS BELOW CAREFULLY.
IN PARTICULAR, MAKE SURE TO ALWAYS SUBMIT BOTH THE 2D BOUNDING BOXES AND THE
3D BOUNDING BOXES AND FILTER BOUNDING BOXES NOT VISIBLE ON THE IMAGE PLANE.

Data Format Description
=======================

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/ contains the left color camera images (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text file)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m, writeLabels.m) provided within
this devkit. All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

Here, 'DontCare' labels denote regions in which objects have not been labeled,
for example because they have been too far away from the laser scanner. To
prevent such objects from being counted as false positives our evaluation
script will ignore objects detected in don't care regions of the test set.
You can use the don't care labels in the training set to avoid that your object
detector is harvesting hard negatives from those areas, in case you consider
non-object regions from the training images as negative examples.

The coordinates in the camera coordinate system can be projected in the image
by using the 3x4 projection matrix in the calib folder, where for the left
color camera for which the images are provided, P2 must be used. The
difference between rotation_y and alpha is, that rotation_y is directly
given in camera coordinates, while alpha also considers the vector from the
camera center to the object center, to compute the relative orientation of
the object with respect to the camera. For example, a car which is facing
along the X-axis of the camera coordinate system corresponds to rotation_y=0,
no matter where it is located in the X/Z plane (bird's eye view), while
alpha is zero only, when this object is located along the Z-axis of the
camera. When moving the car away from the Z-axis, the observation angle
will change.

To project a point from Velodyne coordinates into the left color image,
you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y
For the right color image: x = P3 * R0_rect * Tr_velo_to_cam * y

Note: All matrices are stored row-major, i.e., the first values correspond
to the first row. R0_rect contains a 3x3 matrix which you need to extend to
a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix 
in the same way!

Note, that while all this information is available for the training data,
only the data which is actually needed for the particular benchmark must
be provided to the evaluation server. However, all 15 values must be provided
at all times, with the unused ones set to their default values (=invalid) as
specified in writeLabels.m. Additionally a 16'th value must be provided
with a floating value of the score for a particular detection, where higher
indicates higher confidence in the detection. The range of your scores will
be automatically determined by our evaluation server, you don't have to
normalize it, but it should be roughly linear. If you use writeLabels.m for
writing your results, this function will take care of storing all required
data correctly.

2D Object Detection Benchmark
=============================

The goal in the 2D object detection task is to train object detectors for the
classes 'Car', 'Pedestrian', and 'Cyclist'. The object detectors must
provide as output the 2D 0-based bounding box in the image using the format
specified above, as well as a detection score, indicating the confidence
in the detection. All other values must be set to their default values
(=invalid), see above. One text file per image must be provided in a zip
archive, where each file can contain many detections, depending on the 
number of objects per image. In our evaluation we only evaluate detections/
objects larger than 25 pixel (height) in the image and do not count 'Van' as
false positives for 'Car' or 'Sitting Person' as false positive for 'Pedestrian'
due to their similarity in appearance. As evaluation criterion we follow
PASCAL and require the intersection-over-union of bounding boxes to be
larger than 50% for an object to be detected correctly.

Object Orientation Estimation Benchmark
=======================================

This benchmark is similar as the previous one, except that you have to
provide additionally the most likely relative object observation angle
(=alpha) for each detection. As described in our paper, our score here
considers both, the detection performance as well as the orientation
estimation performance of the algorithm jointly.

3D Object Detection Benchmark
=============================

The goal in the 3D object detection task is to train object detectors for
the classes 'Car', 'Pedestrian', and 'Cyclist'. The object detectors
must provide BOTH the 2D 0-based bounding box in the image as well as the 3D
bounding box (in the format specified above, i.e. 3D dimensions and 3D locations)
and the detection score/confidence. Note that the 2D bounding box should correspond
to the projection of the 3D bounding box - this is required to filter objects
larger than 25 pixel (height). We also note that not all objects in the point clouds
have been labeled. To avoid false positives, detections not visible on the image plane
should be filtered (the evaluation does not take care of this, see 
'cpp/evaluate_object.cpp'). Similar to the 2D object detection benchmark,
we do not count 'Van' as false positives for 'Car' or 'Sitting Person'
as false positive for 'Pedestrian'. Evaluation criterion follows the 2D
object detection benchmark (using 3D bounding box overlap).

Bird's Eye View Benchmark
=========================

The goal in the bird's eye view detection task is to train object detectors
for the classes 'Car', 'Pedestrian', and 'Cyclist' where the detectors must provide
BOTH the 2D 0-based bounding box in the image as well as the 3D bounding box
in bird's eye view and the detection score/confidence. This means that the 3D
bounding box does not have to include information on the height axis, i.e.
the height of the bounding box and the bounding box location along the height axis.
For example, when evaluating the bird's eye view benchmark only (without the
3D object detection benchmark), the height of the bounding box can be set to
a value equal to or smaller than zero. Similarly, the y-axis location of the
bounding box can be set to -1000 (note that an arbitrary negative value will
not work). As above, we note that the 2D bounding boxes are required to filter
objects larger than 25 pixel (height) and that - to avoid false positives - detections
not visible on the image plane should be filtered. As in all benchmarks, we do
not count 'Van' as false positives for 'Car' or 'Sitting Person' as false positive
for 'Pedestrian'. Evaluation criterion follows the above benchmarks using
a bird's eye view bounding box overlap.

Mapping to Raw Data
===================

Note that this section is additional to the benchmark, and not required for
solving the object detection task.

In order to allow the usage of the laser point clouds, gps data, the right
camera image and the grayscale images for the TRAINING data as well, we
provide the mapping of the training set to the raw data of the KITTI dataset.

This information is saved in mapping/train_mapping.txt and train_rand.txt:

train_rand.txt: Random permutation, assigning a unique index to each image
from the object detection training set. The index is 1-based.

train_mapping.txt: Maps each unique index (= 1-based line numbers) to a zip
file of the KITTI raw data set files. Note that those files are split into
several categories on the website!

Example: Image 0 from the training set has index 7282 and maps to date
2011_09_28, drive 106 and frame 48. Drives and frames are 0-based.

Evaluation Protocol:
====================

For transparency we have included the KITTI evaluation code in the
subfolder 'cpp' of this development kit. It can be compiled via:

g++ -O3 -DNDEBUG -o evaluate_object evaluate_object.cpp

or using CMake and the provided 'CMakeLists.txt'.

IMPORTANT NOTE:

This code will result in 41 values (41 recall discretization steps). Following the MonoDIS paper

https://research.mapillary.com/img/publications/MonoDIS.pdf

from 8.10.2019 we compute the average precision not like in the PASCAL VOC protocol, but as follows:

sum = 0;
for (i=1; i<=40; i++)
  sum += vals[i];
average = sum/40.0;
'''