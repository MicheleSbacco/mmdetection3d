# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from os import path as osp

import mmcv
import mmengine
import numpy as np
from mmcv.ops import roi_align
from mmdet.evaluation import bbox_overlaps
from mmengine import print_log, track_iter_progress
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops


def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair
    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks).to(device).index_select(
            0, pos_assigned_gt_inds).to(dtype=rois.dtype))
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = (
        roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1))
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks



# Copied function from "create_groundtruth_database"
# Peculiarities:
#   - Suppresses all the unused arguments("relative_path", "add_rgb", "lidar_only", "bev_only", "coors_range")
#   - Suppresses the mask-related arguments ("mask_anno_path", "with_mask")
def create_michele_custom_groundtruth_database(dataset_class_name,
                                               data_path,
                                               info_prefix,
                                               info_path=None,
                                               used_classes=None,
                                               database_save_path=None,
                                               db_info_save_path=None):

    # Just a warning print
    print(f'Create GT Database of {dataset_class_name}')

    # Create a dataset configuration that will be used to build a dataset.
    #   - If need to use images, just make really similar to Kitti (just some peculiarities as
    #     explained above)
    #   - Otherwise, use the custom dataset in "mmdet3d/datasets"
    dataset_cfg = dict(
        type=dataset_class_name,
        data_root=data_path, 
        ann_file=info_path,
        modality=dict(
            use_lidar=True,
            use_camera=False,                                   ##  With kitti was enabled just in case of segmentation mask. Probably
                                                                #   for fusion I will need it True
        ),
        data_prefix=dict(
            pts='training/velodyne_reduced'
        ),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                backend_args=None)
        ]
    )
    # Update the configuration if need to use images (essentially, just make it same as KITTI preparation)
    if "Kitti" in dataset_class_name:
        dataset_cfg.update(
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'
            ),
            modality=dict(
                use_lidar=True,
                use_camera=False,
            )
        )
    # Build the dataset with mmengine
    dataset = DATASETS.build(dataset_cfg)

    # Create names for:
    #   - GT Database (instance by instance)
    #   - GT Pickle File
    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    # Variable needed to track "categories" in case they are not made explicit              ##  In our case we don't make them explicit, but if
                                                                                            #   needed can be passed to the function with the 
                                                                                            #   argument "used_classes"
    group_counter = 0
    # For loop: does the "images" (or scans) one by one
    for j in track_iter_progress(list(range(len(dataset)))):
        
        # Differences between "data_info" and "example" (from ChatGPT):
        #   - Second one has less repetitions
        #   - Second one has directly the point cloud in the form of a tensor
        #   - Both have some data from "calib"!!!
        #   - Examples at the end of function
        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        
        # Get some various info from the "example" dictionary
        annos = example['ann_info']
        sample_idx = example['sample_idx']                                                   ##  Not really an image, but the sample number (also 
                                                                                            #   present in the point-cloud after modifications)
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        # This is the dictionary that stores the group types if "used_classes" argument is None
        group_dict = dict()
        # If "group_ids" not present in "annos" (in our case, NOT present) there is a "group_id" for each singe bbox in the image/scan
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)

        # Take the number of bboxes, and points in the point cloud for each bbox
        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
        # For loop: does one by one the bboxes of the single image/scan (loop in the loop)
        for i in range(num_obj):
            filename = f'{sample_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': sample_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)



def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        backend_args = None
        dataset_cfg.update(
            modality=dict(
                use_lidar=True,
                use_camera=with_mask,
            ),
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        backend_args = None
        dataset_cfg.update(
            test_mode=False,
            data_prefix=dict(
                pts='training/velodyne', img='', sweeps='training/velodyne'),
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=6,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info['file_name']: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in file2id.keys():
                print(f'skip image {img_path} for empty mask')
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


class GTDatabaseCreater:
    """Given the raw data, generate the ground truth database. This is the
    parallel version. For serialized version, please refer to
    `create_groundtruth_database`

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
        num_worker (int, optional): the number of parallel workers to use.
            Default: 8.
    """

    def __init__(self,
                 dataset_class_name,
                 data_path,
                 info_prefix,
                 info_path=None,
                 mask_anno_path=None,
                 used_classes=None,
                 database_save_path=None,
                 db_info_save_path=None,
                 relative_path=True,
                 add_rgb=False,
                 lidar_only=False,
                 bev_only=False,
                 coors_range=None,
                 with_mask=False,
                 num_worker=8) -> None:
        self.dataset_class_name = dataset_class_name
        self.data_path = data_path
        self.info_prefix = info_prefix
        self.info_path = info_path
        self.mask_anno_path = mask_anno_path
        self.used_classes = used_classes
        self.database_save_path = database_save_path
        self.db_info_save_path = db_info_save_path
        self.relative_path = relative_path
        self.add_rgb = add_rgb
        self.lidar_only = lidar_only
        self.bev_only = bev_only
        self.coors_range = coors_range
        self.with_mask = with_mask
        self.num_worker = num_worker
        self.pipeline = None

    def create_single(self, input_dict):
        group_counter = 0
        single_db_infos = dict()
        example = self.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [
            self.dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']
        ]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if self.with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in self.file2id.keys():
                print(f'skip image {img_path} for empty mask')
                return single_db_infos
            img_id = self.file2id[img_path]
            kins_annIds = self.coco.getAnnIds(imgIds=img_id)
            kins_raw_info = self.coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(self.database_save_path, filename)
            rel_filepath = osp.join(f'{self.info_prefix}_gt_database',
                                    filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if self.with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (self.used_classes is None) or names[i] in self.used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if self.with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in single_db_infos:
                    single_db_infos[names[i]].append(db_info)
                else:
                    single_db_infos[names[i]] = [db_info]

        return single_db_infos

    def create(self):
        print_log(
            f'Create GT Database of {self.dataset_class_name}',
            logger='current')
        dataset_cfg = dict(
            type=self.dataset_class_name,
            data_root=self.data_path,
            ann_file=self.info_path)
        if self.dataset_class_name == 'KittiDataset':
            backend_args = None
            dataset_cfg.update(
                test_mode=False,
                data_prefix=dict(
                    pts='training/velodyne_reduced', img='training/image_2'),
                modality=dict(
                    use_lidar=True,
                    use_depth=False,
                    use_lidar_intensity=True,
                    use_camera=self.with_mask,
                ),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=4,
                        use_dim=4,
                        backend_args=backend_args),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True,
                        backend_args=backend_args)
                ])

        elif self.dataset_class_name == 'NuScenesDataset':
            dataset_cfg.update(
                use_valid_flag=True,
                data_prefix=dict(
                    pts='samples/LIDAR_TOP', img='',
                    sweeps='sweeps/LIDAR_TOP'),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5),
                    dict(
                        type='LoadPointsFromMultiSweeps',
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True)
                ])

        elif self.dataset_class_name == 'WaymoDataset':
            backend_args = None
            dataset_cfg.update(
                test_mode=False,
                data_prefix=dict(
                    pts='training/velodyne',
                    img='',
                    sweeps='training/velodyne'),
                modality=dict(
                    use_lidar=True,
                    use_depth=False,
                    use_lidar_intensity=True,
                    use_camera=False,
                ),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=6,
                        use_dim=6,
                        backend_args=backend_args),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True,
                        backend_args=backend_args)
                ])

        self.dataset = DATASETS.build(dataset_cfg)
        self.pipeline = self.dataset.pipeline
        if self.database_save_path is None:
            self.database_save_path = osp.join(
                self.data_path, f'{self.info_prefix}_gt_database')
        if self.db_info_save_path is None:
            self.db_info_save_path = osp.join(
                self.data_path, f'{self.info_prefix}_dbinfos_train.pkl')
        mmengine.mkdir_or_exist(self.database_save_path)
        if self.with_mask:
            self.coco = COCO(osp.join(self.data_path, self.mask_anno_path))
            imgIds = self.coco.getImgIds()
            self.file2id = dict()
            for i in imgIds:
                info = self.coco.loadImgs([i])[0]
                self.file2id.update({info['file_name']: i})

        def loop_dataset(i):
            input_dict = self.dataset.get_data_info(i)
            input_dict['box_type_3d'] = self.dataset.box_type_3d
            input_dict['box_mode_3d'] = self.dataset.box_mode_3d
            return input_dict

        if self.num_worker == 0:
            multi_db_infos = mmengine.track_progress(
                self.create_single,
                ((loop_dataset(i)
                  for i in range(len(self.dataset))), len(self.dataset)))
        else:
            multi_db_infos = mmengine.track_parallel_progress(
                self.create_single,
                ((loop_dataset(i)
                  for i in range(len(self.dataset))), len(self.dataset)),
                self.num_worker,
                chunksize=1000)
        print_log('Make global unique group id', logger='current')
        group_counter_offset = 0
        all_db_infos = dict()
        for single_db_infos in track_iter_progress(multi_db_infos):
            group_id = -1
            for name, name_db_infos in single_db_infos.items():
                for db_info in name_db_infos:
                    group_id = max(group_id, db_info['group_id'])
                    db_info['group_id'] += group_counter_offset
                if name not in all_db_infos:
                    all_db_infos[name] = []
                all_db_infos[name].extend(name_db_infos)
            group_counter_offset += (group_id + 1)

        for k, v in all_db_infos.items():
            print_log(f'load {len(v)} {k} database infos', logger='current')

        print_log(f'Saving GT database infos into {self.db_info_save_path}')
        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)



'''
"data_info" iteration 0:
{'sample_idx': 0, 'images': {'CAM0': {'cam2img': [[707.0493, 0.0, 604.0814, 0.0], [0.0, 707.0493, 180.5066, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -216.70103429706526], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.22322079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]]}, 'CAM1': {'cam2img': [[707.0493, 0.0, 604.0814, -379.7842], [0.0, 707.0493, 180.5066, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -596.4852342970653], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.22322079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]]}, 'CAM2': {'img_path': '000000.png', 'height': 370, 'width': 1224, 'cam2img': [[707.0493, 0.0, 604.0814, 45.75831], [0.0, 707.0493, 180.5066, -0.3454157], [0.0, 0.0, 1.0, 0.004981016], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -170.94272429706527], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.56863649623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.3275679897468414], [0.0, 0.0, 0.0, 1.0]], 'lidar2cam': [[-0.0015960992313921452, -0.9999162554740906, -0.012840436771512032, -0.022366708144545555], [-0.00527064548805356, 0.012848696671426296, -0.9999035596847534, -0.05967890843749046], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'cam2img': [[707.0493, 0.0, 604.0814, -334.1081], [0.0, 707.0493, 180.5066, 2.33066], [0.0, 0.0, 1.0, 0.003201153], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -550.8091342970653], [176.77725024402156, 8.80879965630006, -707.9361204188122, -99.89256079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.3293478527468414], [0.0, 0.0, 0.0, 1.0]]}, 'R0_rect': [[0.9999127984046936, 0.010092630051076412, -0.008511931635439396, 0.0], [-0.010127290152013302, 0.9999405741691589, -0.004037670791149139, 0.0], [0.008470674976706505, 0.0041235219687223434, 0.9999555945396423, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000000.bin', 'Tr_velo_to_cam': [[0.006927963811904192, -0.9999722242355347, -0.0027578289154917, -0.024577289819717407], [-0.0011629819637164474, 0.0027498360723257065, -0.9999955296516418, -0.06127237156033516], [0.999975323677063, 0.006931141018867493, -0.0011438990477472544, -0.33210289478302], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'instances': [{'bbox': [712.4, 143.0, 810.73, 307.92], 'bbox_label': 0, 'bbox_3d': [1.84, 1.47, 8.41, 1.2, 1.89, 0.48, 0.01], 'bbox_label_3d': 0, 'depth': 8.4149808883667, 'center_2d': [763.7633056640625, 224.4706268310547], 'num_lidar_pts': 377, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -0.2, 'score': 0.0, 'index': 0, 'group_id': 0}], 'cam_instances': {'CAM2': [{'bbox_label': 0, 'bbox_label_3d': 0, 'bbox': [710.4446301035068, 144.00207112943306, 820.2930685018162, 307.58688675239017], 'bbox_3d_isvalid': True, 'bbox_3d': [1.840000033378601, 1.4700000286102295, 8.40999984741211, 1.2000000476837158, 1.8899999856948853, 0.47999998927116394, 0.009999999776482582], 'velocity': -1, 'center_2d': [763.7633056640625, 224.4706268310547], 'depth': 8.4149808883667}]}, 'plane': None, 'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000000.bin', 'ann_info': {'gt_bboxes': array([[712.4 , 143.  , 810.73, 307.92]], dtype=float32), 'gt_bboxes_labels': array([0]), 'gt_bboxes_3d': LiDARInstance3DBoxes(
    tensor([[ 8.7314, -1.8559, -1.5997,  1.2000,  0.4800,  1.8900, -1.5808]])), 'gt_labels_3d': array([0]), 'depths': array([8.414981], dtype=float32), 'centers_2d': array([[763.7633 , 224.47063]], dtype=float32), 'num_lidar_pts': array([377]), 'difficulty': array([0]), 'truncated': array([0.]), 'occluded': array([0]), 'alpha': array([-0.2]), 'score': array([0.]), 'index': array([0]), 'group_id': array([0]), 'instances': [{'bbox': [712.4, 143.0, 810.73, 307.92], 'bbox_label': 0, 'bbox_3d': [1.84, 1.47, 8.41, 1.2, 1.89, 0.48, 0.01], 'bbox_label_3d': 0, 'depth': 8.4149808883667, 'center_2d': [763.7633056640625, 224.4706268310547], 'num_lidar_pts': 377, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -0.2, 'score': 0.0, 'index': 0, 'group_id': 0}]}}

"example" iteration 0:
{'sample_idx': 0, 'images': {'CAM0': {'cam2img': [[707.0493, 0.0, 604.0814, 0.0], [0.0, 707.0493, 180.5066, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -216.70103429706526], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.22322079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]]}, 'CAM1': {'cam2img': [[707.0493, 0.0, 604.0814, -379.7842], [0.0, 707.0493, 180.5066, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -596.4852342970653], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.22322079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]]}, 'CAM2': {'img_path': '000000.png', 'height': 370, 'width': 1224, 'cam2img': [[707.0493, 0.0, 604.0814, 45.75831], [0.0, 707.0493, 180.5066, -0.3454157], [0.0, 0.0, 1.0, 0.004981016], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -170.94272429706527], [176.77725024402156, 8.80879965630006, -707.9361204188122, -102.56863649623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.3275679897468414], [0.0, 0.0, 0.0, 1.0]], 'lidar2cam': [[-0.0015960992313921452, -0.9999162554740906, -0.012840436771512032, -0.022366708144545555], [-0.00527064548805356, 0.012848696671426296, -0.9999035596847534, -0.05967890843749046], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.33254900574684143], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'cam2img': [[707.0493, 0.0, 604.0814, -334.1081], [0.0, 707.0493, 180.5066, 2.33066], [0.0, 0.0, 1.0, 0.003201153], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[602.9436976111118, -707.9132863314612, -12.274842681831494, -550.8091342970653], [176.77725024402156, 8.80879965630006, -707.9361204188122, -99.89256079623453], [0.9999848008155823, -0.0015282672829926014, -0.005290712229907513, -0.3293478527468414], [0.0, 0.0, 0.0, 1.0]]}, 'R0_rect': [[0.9999127984046936, 0.010092630051076412, -0.008511931635439396, 0.0], [-0.010127290152013302, 0.9999405741691589, -0.004037670791149139, 0.0], [0.008470674976706505, 0.0041235219687223434, 0.9999555945396423, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000000.bin', 'Tr_velo_to_cam': [[0.006927963811904192, -0.9999722242355347, -0.0027578289154917, -0.024577289819717407], [-0.0011629819637164474, 0.0027498360723257065, -0.9999955296516418, -0.06127237156033516], [0.999975323677063, 0.006931141018867493, -0.0011438990477472544, -0.33210289478302], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'instances': [{'bbox': [712.4, 143.0, 810.73, 307.92], 'bbox_label': 0, 'bbox_3d': [1.84, 1.47, 8.41, 1.2, 1.89, 0.48, 0.01], 'bbox_label_3d': 0, 'depth': 8.4149808883667, 'center_2d': [763.7633056640625, 224.4706268310547], 'num_lidar_pts': 377, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -0.2, 'score': 0.0, 'index': 0, 'group_id': 0}], 'cam_instances': {'CAM2': [{'bbox_label': 0, 'bbox_label_3d': 0, 'bbox': [710.4446301035068, 144.00207112943306, 820.2930685018162, 307.58688675239017], 'bbox_3d_isvalid': True, 'bbox_3d': [1.840000033378601, 1.4700000286102295, 8.40999984741211, 1.2000000476837158, 1.8899999856948853, 0.47999998927116394, 0.009999999776482582], 'velocity': -1, 'center_2d': [763.7633056640625, 224.4706268310547], 'depth': 8.4149808883667}]}, 'plane': None, 'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000000.bin', 'ann_info': {'gt_bboxes': array([[712.4 , 143.  , 810.73, 307.92]], dtype=float32), 'gt_bboxes_labels': array([0]), 'gt_bboxes_3d': LiDARInstance3DBoxes(
    tensor([[ 8.7314, -1.8559, -1.5997,  1.2000,  0.4800,  1.8900, -1.5808]])), 'gt_labels_3d': array([0]), 'depths': array([8.414981], dtype=float32), 'centers_2d': array([[763.7633 , 224.47063]], dtype=float32), 'num_lidar_pts': array([377]), 'difficulty': array([0]), 'truncated': array([0.]), 'occluded': array([0]), 'alpha': array([-0.2]), 'score': array([0.]), 'index': array([0]), 'group_id': array([0]), 'instances': [{'bbox': [712.4, 143.0, 810.73, 307.92], 'bbox_label': 0, 'bbox_3d': [1.84, 1.47, 8.41, 1.2, 1.89, 0.48, 0.01], 'bbox_label_3d': 0, 'depth': 8.4149808883667, 'center_2d': [763.7633056640625, 224.4706268310547], 'num_lidar_pts': 377, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -0.2, 'score': 0.0, 'index': 0, 'group_id': 0}]}, 'points': LiDARPoints(
    tensor([[ 1.8324e+01,  4.9000e-02,  8.2900e-01,  0.0000e+00],
        [ 1.8344e+01,  1.0600e-01,  8.2900e-01,  0.0000e+00],
        [ 5.1299e+01,  5.0500e-01,  1.9440e+00,  0.0000e+00],
        ...,
        [ 6.2720e+00, -4.0000e-02, -1.6370e+00,  3.0000e-01],
        [ 6.2740e+00, -3.1000e-02, -1.6370e+00,  3.1000e-01],
        [ 6.2760e+00, -1.1000e-02, -1.6380e+00,  3.1000e-01]])), 'gt_bboxes_3d': LiDARInstance3DBoxes(
    tensor([[ 8.7314, -1.8559, -1.5997,  1.2000,  0.4800,  1.8900, -1.5808]])), 'gt_labels_3d': array([0])}


    
"data_info" iteration 2: (not 1)
{'sample_idx': 2, 'images': {'CAM0': {'cam2img': [[721.5377, 0.0, 609.5593, 0.0], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -167.8990963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM1': {'cam2img': [[721.5377, 0.0, 609.5593, -387.5744], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -555.4734963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM2': {'img_path': '000007.png', 'height': 375, 'width': 1242, 'cam2img': [[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -123.04181637996919], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.01668911726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2693869299305115], [0.0, 0.0, 0.0, 1.0]], 'lidar2cam': [[0.00023477392096538097, -0.9999441504478455, -0.01056347694247961, -0.002796816872432828], [0.010449407622218132, 0.010565354488790035, -0.999889612197876, -0.07510878890752792], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'cam2img': [[721.5377, 0.0, 609.5593, -339.5242], [0.0, 721.5377, 172.854, 2.199936], [0.0, 0.0, 1.0, 0.002729905], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -507.4232963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -99.03313221726785], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2694029089305115], [0.0, 0.0, 0.0, 1.0]]}, 'R0_rect': [[0.9999238848686218, 0.009837759658694267, -0.007445048075169325, 0.0], [-0.00986979529261589, 0.9999421238899231, -0.004278459120541811, 0.0], [0.007402527146041393, 0.0043516140431165695, 0.999963104724884, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000007.bin', 'Tr_velo_to_cam': [[0.0075337449088692665, -0.9999713897705078, -0.00061660201754421, -0.004069766029715538], [0.01480249036103487, 0.0007280732970684767, -0.9998902082443237, -0.07631617784500122], [0.9998620748519897, 0.007523790001869202, 0.014807550236582756, -0.2717806100845337], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'instances': [{'bbox': [564.62, 174.59, 616.43, 224.74], 'bbox_label': 2, 'bbox_3d': [-0.69, 1.69, 25.01, 3.2, 1.61, 1.66, -1.59], 'bbox_label_3d': 2, 'depth': 25.012746810913086, 'center_2d': [591.3814697265625, 198.3730926513672], 'num_lidar_pts': 182, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -1.56, 'score': 0.0, 'index': 0, 'group_id': 0}, {'bbox': [481.59, 180.09, 512.55, 202.42], 'bbox_label': 2, 'bbox_3d': [-7.43, 1.88, 47.55, 3.7, 1.4, 1.51, 1.55], 'bbox_label_3d': 2, 'depth': 47.5527458190918, 'center_2d': [497.7289123535156, 190.75320434570312], 'num_lidar_pts': 20, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.71, 'score': 0.0, 'index': 1, 'group_id': 1}, {'bbox': [542.05, 175.55, 565.27, 193.79], 'bbox_label': 2, 'bbox_3d': [-4.71, 1.71, 60.52, 4.05, 1.46, 1.66, 1.56], 'bbox_label_3d': 2, 'depth': 60.52274703979492, 'center_2d': [554.121337890625, 184.53306579589844], 'num_lidar_pts': 5, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.64, 'score': 0.0, 'index': 2, 'group_id': 2}, {'bbox': [330.6, 176.09, 355.61, 213.6], 'bbox_label': 1, 'bbox_3d': [-12.63, 1.88, 34.09, 1.95, 1.72, 0.5, 1.54], 'bbox_label_3d': 1, 'depth': 34.09274673461914, 'center_2d': [343.5250549316406, 194.4336700439453], 'num_lidar_pts': 25, 'difficulty': 1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.89, 'score': 0.0, 'index': 3, 'group_id': 3}, {'bbox': [753.33, 164.32, 798.0, 186.74], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 4}, {'bbox': [738.5, 171.32, 753.27, 184.42], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 5}], 'cam_instances': {'CAM2': [{'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [565.4822720402807, 175.01202566042497, 616.6555088322534, 224.96047091220345], 'bbox_3d_isvalid': True, 'bbox_3d': [-0.6899999976158142, 1.690000057220459, 25.010000228881836, 3.200000047683716, 1.6100000143051147, 1.659999966621399, -1.590000033378601], 'velocity': -1, 'center_2d': [591.3814697265625, 198.3730926513672], 'depth': 25.012746810913086}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [481.8496708488522, 179.85710612050596, 512.4094377621442, 202.53901525985071], 'bbox_3d_isvalid': True, 'bbox_3d': [-7.429999828338623, 1.8799999952316284, 47.54999923706055, 3.700000047683716, 1.399999976158142, 1.5099999904632568, 1.5499999523162842], 'velocity': -1, 'center_2d': [497.7289123535156, 190.75320434570312], 'depth': 47.5527458190918}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [542.2247151650495, 175.73341152322814, 565.2443490828854, 193.9446887816074], 'bbox_3d_isvalid': True, 'bbox_3d': [-4.710000038146973, 1.7100000381469727, 60.52000045776367, 4.050000190734863, 1.4600000381469727, 1.659999966621399, 1.559999942779541], 'velocity': -1, 'center_2d': [554.121337890625, 184.53306579589844], 'depth': 60.52274703979492}, {'bbox_label': 1, 'bbox_label_3d': 1, 'bbox': [330.84191493374504, 176.13804311926262, 355.4978537323491, 213.8147876869614], 'bbox_3d_isvalid': True, 'bbox_3d': [-12.630000114440918, 1.8799999952316284, 34.09000015258789, 1.9500000476837158, 1.7200000286102295, 0.5, 1.5399999618530273], 'velocity': -1, 'center_2d': [343.5250549316406, 194.4336700439453], 'depth': 34.09274673461914}]}, 'plane': None, 'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000007.bin', 'ann_info': {'gt_bboxes': array([[564.62, 174.59, 616.43, 224.74],
       [481.59, 180.09, 512.55, 202.42],
       [542.05, 175.55, 565.27, 193.79],
       [330.6 , 176.09, 355.61, 213.6 ]], dtype=float32), 'gt_bboxes_labels': array([2, 2, 2, 1]), 'gt_bboxes_3d': LiDARInstance3DBoxes(
    tensor([[ 2.5299e+01,  7.0896e-01, -1.4934e+00,  3.2000e+00,  1.6600e+00,
          1.6100e+00,  1.9204e-02],
        [ 4.7838e+01,  7.4534e+00, -1.3766e+00,  3.7000e+00,  1.5100e+00,
          1.4000e+00, -3.1208e+00],
        [ 6.0806e+01,  4.7334e+00, -1.0998e+00,  4.0500e+00,  1.6600e+00,
          1.4600e+00, -3.1308e+00],
        [ 3.4378e+01,  1.2651e+01, -1.4624e+00,  1.9500e+00,  5.0000e-01,
          1.7200e+00, -3.1108e+00]])), 'gt_labels_3d': array([2, 2, 2, 1]), 'depths': array([25.012747, 47.552746, 60.522747, 34.092747], dtype=float32), 'centers_2d': array([[591.3815 , 198.3731 ],
       [497.7289 , 190.7532 ],
       [554.12134, 184.53307],
       [343.52505, 194.43367]], dtype=float32), 'num_lidar_pts': array([182,  20,   5,  25]), 'difficulty': array([ 0, -1, -1,  1]), 'truncated': array([0., 0., 0., 0.]), 'occluded': array([0, 0, 0, 0]), 'alpha': array([-1.56,  1.71,  1.64,  1.89]), 'score': array([0., 0., 0., 0.]), 'index': array([0, 1, 2, 3]), 'group_id': array([0, 1, 2, 3]), 'instances': [{'bbox': [564.62, 174.59, 616.43, 224.74], 'bbox_label': 2, 'bbox_3d': [-0.69, 1.69, 25.01, 3.2, 1.61, 1.66, -1.59], 'bbox_label_3d': 2, 'depth': 25.012746810913086, 'center_2d': [591.3814697265625, 198.3730926513672], 'num_lidar_pts': 182, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -1.56, 'score': 0.0, 'index': 0, 'group_id': 0}, {'bbox': [481.59, 180.09, 512.55, 202.42], 'bbox_label': 2, 'bbox_3d': [-7.43, 1.88, 47.55, 3.7, 1.4, 1.51, 1.55], 'bbox_label_3d': 2, 'depth': 47.5527458190918, 'center_2d': [497.7289123535156, 190.75320434570312], 'num_lidar_pts': 20, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.71, 'score': 0.0, 'index': 1, 'group_id': 1}, {'bbox': [542.05, 175.55, 565.27, 193.79], 'bbox_label': 2, 'bbox_3d': [-4.71, 1.71, 60.52, 4.05, 1.46, 1.66, 1.56], 'bbox_label_3d': 2, 'depth': 60.52274703979492, 'center_2d': [554.121337890625, 184.53306579589844], 'num_lidar_pts': 5, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.64, 'score': 0.0, 'index': 2, 'group_id': 2}, {'bbox': [330.6, 176.09, 355.61, 213.6], 'bbox_label': 1, 'bbox_3d': [-12.63, 1.88, 34.09, 1.95, 1.72, 0.5, 1.54], 'bbox_label_3d': 1, 'depth': 34.09274673461914, 'center_2d': [343.5250549316406, 194.4336700439453], 'num_lidar_pts': 25, 'difficulty': 1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.89, 'score': 0.0, 'index': 3, 'group_id': 3}, {'bbox': [753.33, 164.32, 798.0, 186.74], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 4}, {'bbox': [738.5, 171.32, 753.27, 184.42], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 5}]}}

"example" iteration 2: (not 1)
{'sample_idx': 2, 'images': {'CAM0': {'cam2img': [[721.5377, 0.0, 609.5593, 0.0], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -167.8990963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM1': {'cam2img': [[721.5377, 0.0, 609.5593, -387.5744], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -555.4734963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.23306821726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM2': {'img_path': '000007.png', 'height': 375, 'width': 1242, 'cam2img': [[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -123.04181637996919], [180.384193781635, 7.64479865145192, -719.6515015339527, -101.01668911726784], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2693869299305115], [0.0, 0.0, 0.0, 1.0]], 'lidar2cam': [[0.00023477392096538097, -0.9999441504478455, -0.01056347694247961, -0.002796816872432828], [0.010449407622218132, 0.010565354488790035, -0.999889612197876, -0.07510878890752792], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2721328139305115], [0.0, 0.0, 0.0, 1.0]]}, 'CAM3': {'cam2img': [[721.5377, 0.0, 609.5593, -339.5242], [0.0, 721.5377, 172.854, 2.199936], [0.0, 0.0, 1.0, 0.002729905], [0.0, 0.0, 0.0, 1.0]], 'lidar2img': [[609.6953812723476, -721.4215942962135, -1.2512579994207245, -507.4232963799692], [180.384193781635, 7.64479865145192, -719.6515015339527, -99.03313221726785], [0.999945342540741, 0.00012436549877747893, 0.010451302863657475, -0.2694029089305115], [0.0, 0.0, 0.0, 1.0]]}, 'R0_rect': [[0.9999238848686218, 0.009837759658694267, -0.007445048075169325, 0.0], [-0.00986979529261589, 0.9999421238899231, -0.004278459120541811, 0.0], [0.007402527146041393, 0.0043516140431165695, 0.999963104724884, 0.0], [0.0, 0.0, 0.0, 1.0]]}, 'lidar_points': {'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000007.bin', 'Tr_velo_to_cam': [[0.0075337449088692665, -0.9999713897705078, -0.00061660201754421, -0.004069766029715538], [0.01480249036103487, 0.0007280732970684767, -0.9998902082443237, -0.07631617784500122], [0.9998620748519897, 0.007523790001869202, 0.014807550236582756, -0.2717806100845337], [0.0, 0.0, 0.0, 1.0]], 'Tr_imu_to_velo': [[0.999997615814209, 0.0007553070900030434, -0.002035825978964567, -0.8086758852005005], [-0.0007854027207940817, 0.9998897910118103, -0.014822980388998985, 0.3195559084415436], [0.002024406101554632, 0.014824540354311466, 0.9998881220817566, -0.7997230887413025], [0.0, 0.0, 0.0, 1.0]]}, 'instances': [{'bbox': [564.62, 174.59, 616.43, 224.74], 'bbox_label': 2, 'bbox_3d': [-0.69, 1.69, 25.01, 3.2, 1.61, 1.66, -1.59], 'bbox_label_3d': 2, 'depth': 25.012746810913086, 'center_2d': [591.3814697265625, 198.3730926513672], 'num_lidar_pts': 182, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -1.56, 'score': 0.0, 'index': 0, 'group_id': 0}, {'bbox': [481.59, 180.09, 512.55, 202.42], 'bbox_label': 2, 'bbox_3d': [-7.43, 1.88, 47.55, 3.7, 1.4, 1.51, 1.55], 'bbox_label_3d': 2, 'depth': 47.5527458190918, 'center_2d': [497.7289123535156, 190.75320434570312], 'num_lidar_pts': 20, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.71, 'score': 0.0, 'index': 1, 'group_id': 1}, {'bbox': [542.05, 175.55, 565.27, 193.79], 'bbox_label': 2, 'bbox_3d': [-4.71, 1.71, 60.52, 4.05, 1.46, 1.66, 1.56], 'bbox_label_3d': 2, 'depth': 60.52274703979492, 'center_2d': [554.121337890625, 184.53306579589844], 'num_lidar_pts': 5, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.64, 'score': 0.0, 'index': 2, 'group_id': 2}, {'bbox': [330.6, 176.09, 355.61, 213.6], 'bbox_label': 1, 'bbox_3d': [-12.63, 1.88, 34.09, 1.95, 1.72, 0.5, 1.54], 'bbox_label_3d': 1, 'depth': 34.09274673461914, 'center_2d': [343.5250549316406, 194.4336700439453], 'num_lidar_pts': 25, 'difficulty': 1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.89, 'score': 0.0, 'index': 3, 'group_id': 3}, {'bbox': [753.33, 164.32, 798.0, 186.74], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 4}, {'bbox': [738.5, 171.32, 753.27, 184.42], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 5}], 'cam_instances': {'CAM2': [{'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [565.4822720402807, 175.01202566042497, 616.6555088322534, 224.96047091220345], 'bbox_3d_isvalid': True, 'bbox_3d': [-0.6899999976158142, 1.690000057220459, 25.010000228881836, 3.200000047683716, 1.6100000143051147, 1.659999966621399, -1.590000033378601], 'velocity': -1, 'center_2d': [591.3814697265625, 198.3730926513672], 'depth': 25.012746810913086}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [481.8496708488522, 179.85710612050596, 512.4094377621442, 202.53901525985071], 'bbox_3d_isvalid': True, 'bbox_3d': [-7.429999828338623, 1.8799999952316284, 47.54999923706055, 3.700000047683716, 1.399999976158142, 1.5099999904632568, 1.5499999523162842], 'velocity': -1, 'center_2d': [497.7289123535156, 190.75320434570312], 'depth': 47.5527458190918}, {'bbox_label': 2, 'bbox_label_3d': 2, 'bbox': [542.2247151650495, 175.73341152322814, 565.2443490828854, 193.9446887816074], 'bbox_3d_isvalid': True, 'bbox_3d': [-4.710000038146973, 1.7100000381469727, 60.52000045776367, 4.050000190734863, 1.4600000381469727, 1.659999966621399, 1.559999942779541], 'velocity': -1, 'center_2d': [554.121337890625, 184.53306579589844], 'depth': 60.52274703979492}, {'bbox_label': 1, 'bbox_label_3d': 1, 'bbox': [330.84191493374504, 176.13804311926262, 355.4978537323491, 213.8147876869614], 'bbox_3d_isvalid': True, 'bbox_3d': [-12.630000114440918, 1.8799999952316284, 34.09000015258789, 1.9500000476837158, 1.7200000286102295, 0.5, 1.5399999618530273], 'velocity': -1, 'center_2d': [343.5250549316406, 194.4336700439453], 'depth': 34.09274673461914}]}, 'plane': None, 'num_pts_feats': 4, 'lidar_path': './data/kitti/training/velodyne_reduced/000007.bin', 'ann_info': {'gt_bboxes': array([[564.62, 174.59, 616.43, 224.74],
       [481.59, 180.09, 512.55, 202.42],
       [542.05, 175.55, 565.27, 193.79],
       [330.6 , 176.09, 355.61, 213.6 ]], dtype=float32), 'gt_bboxes_labels': array([2, 2, 2, 1]), 'gt_bboxes_3d': LiDARInstance3DBoxes(
    tensor([[ 2.5299e+01,  7.0896e-01, -1.4934e+00,  3.2000e+00,  1.6600e+00,
          1.6100e+00,  1.9204e-02],
        [ 4.7838e+01,  7.4534e+00, -1.3766e+00,  3.7000e+00,  1.5100e+00,
          1.4000e+00, -3.1208e+00],
        [ 6.0806e+01,  4.7334e+00, -1.0998e+00,  4.0500e+00,  1.6600e+00,
          1.4600e+00, -3.1308e+00],
        [ 3.4378e+01,  1.2651e+01, -1.4624e+00,  1.9500e+00,  5.0000e-01,
          1.7200e+00, -3.1108e+00]])), 'gt_labels_3d': array([2, 2, 2, 1]), 'depths': array([25.012747, 47.552746, 60.522747, 34.092747], dtype=float32), 'centers_2d': array([[591.3815 , 198.3731 ],
       [497.7289 , 190.7532 ],
       [554.12134, 184.53307],
       [343.52505, 194.43367]], dtype=float32), 'num_lidar_pts': array([182,  20,   5,  25]), 'difficulty': array([ 0, -1, -1,  1]), 'truncated': array([0., 0., 0., 0.]), 'occluded': array([0, 0, 0, 0]), 'alpha': array([-1.56,  1.71,  1.64,  1.89]), 'score': array([0., 0., 0., 0.]), 'index': array([0, 1, 2, 3]), 'group_id': array([0, 1, 2, 3]), 'instances': [{'bbox': [564.62, 174.59, 616.43, 224.74], 'bbox_label': 2, 'bbox_3d': [-0.69, 1.69, 25.01, 3.2, 1.61, 1.66, -1.59], 'bbox_label_3d': 2, 'depth': 25.012746810913086, 'center_2d': [591.3814697265625, 198.3730926513672], 'num_lidar_pts': 182, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -1.56, 'score': 0.0, 'index': 0, 'group_id': 0}, {'bbox': [481.59, 180.09, 512.55, 202.42], 'bbox_label': 2, 'bbox_3d': [-7.43, 1.88, 47.55, 3.7, 1.4, 1.51, 1.55], 'bbox_label_3d': 2, 'depth': 47.5527458190918, 'center_2d': [497.7289123535156, 190.75320434570312], 'num_lidar_pts': 20, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.71, 'score': 0.0, 'index': 1, 'group_id': 1}, {'bbox': [542.05, 175.55, 565.27, 193.79], 'bbox_label': 2, 'bbox_3d': [-4.71, 1.71, 60.52, 4.05, 1.46, 1.66, 1.56], 'bbox_label_3d': 2, 'depth': 60.52274703979492, 'center_2d': [554.121337890625, 184.53306579589844], 'num_lidar_pts': 5, 'difficulty': -1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.64, 'score': 0.0, 'index': 2, 'group_id': 2}, {'bbox': [330.6, 176.09, 355.61, 213.6], 'bbox_label': 1, 'bbox_3d': [-12.63, 1.88, 34.09, 1.95, 1.72, 0.5, 1.54], 'bbox_label_3d': 1, 'depth': 34.09274673461914, 'center_2d': [343.5250549316406, 194.4336700439453], 'num_lidar_pts': 25, 'difficulty': 1, 'truncated': 0.0, 'occluded': 0, 'alpha': 1.89, 'score': 0.0, 'index': 3, 'group_id': 3}, {'bbox': [753.33, 164.32, 798.0, 186.74], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 4}, {'bbox': [738.5, 171.32, 753.27, 184.42], 'bbox_label': -1, 'bbox_3d': [-1000.0, -1000.0, -1000.0, -1.0, -1.0, -1.0, -10.0], 'bbox_label_3d': -1, 'depth': -999.9972534179688, 'center_2d': [1331.055908203125, 894.033203125], 'num_lidar_pts': -1, 'difficulty': -1, 'truncated': -1.0, 'occluded': -1, 'alpha': -10.0, 'score': 0.0, 'index': -1, 'group_id': 5}]}, 'points': LiDARPoints(
    tensor([[ 7.4986e+01,  9.6380e+00,  2.7660e+00,  0.0000e+00],
        [ 7.4286e+01,  9.7850e+00,  2.7430e+00,  0.0000e+00],
        [ 7.3747e+01,  9.8320e+00,  2.7250e+00,  0.0000e+00],
        ...,
        [ 6.4250e+00, -4.2000e-02, -1.6790e+00,  2.2000e-01],
        [ 6.4210e+00, -2.2000e-02, -1.6780e+00,  1.0000e-01],
        [ 6.4250e+00, -2.0000e-03, -1.6790e+00,  2.0000e-01]])), 'gt_bboxes_3d': LiDARInstance3DBoxes(
    tensor([[ 2.5299e+01,  7.0896e-01, -1.4934e+00,  3.2000e+00,  1.6600e+00,
          1.6100e+00,  1.9204e-02],
        [ 4.7838e+01,  7.4534e+00, -1.3766e+00,  3.7000e+00,  1.5100e+00,
          1.4000e+00, -3.1208e+00],
        [ 6.0806e+01,  4.7334e+00, -1.0998e+00,  4.0500e+00,  1.6600e+00,
          1.4600e+00, -3.1308e+00],
        [ 3.4378e+01,  1.2651e+01, -1.4624e+00,  1.9500e+00,  5.0000e-01,
          1.7200e+00, -3.1108e+00]])), 'gt_labels_3d': array([2, 2, 2, 1])}
'''