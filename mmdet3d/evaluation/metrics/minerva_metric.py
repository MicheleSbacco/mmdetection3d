'''
######################################### GENERAL DESCRIPTION ##########################################
'''

# Tried with Dona's version but it's a mess between various files. Will try to make it from scratch to make it lighter
#
# New idea is to use the predictions and manually compute the AP40 with the use o the functions already implemented by
# mmdet3d in the file "mmdet3d/structures/ops/iou3dcalculator.py". 
# The metric would be the 3D-metric (not bbox because it projects on the image plane, does not make sense without any
# image).
# 
# Recap of the analysis for the printing: 
#       - the function "compute_metrics" must return a dictionary of "key[str]: value[float]" but it is printed in the 
#         "ugly" way (like the first block below here)
#       - the function "kitti_evaluate" has been used as a reference for the "beautiful" print, with the specific line 
#         being "print_log(f'Results of " that can be directly integrated inside "compute_metrics"



'''
######################################### SUGGESTION BY CHATGPT ##########################################
'''

# import numpy as np

# def compute_iou_3d(box1, box2):
#     # Calculate the IoU between two 3D bounding boxes
#     # (Implement this function based on the 3D bounding box format)
#     pass

# def match_boxes(pred_bboxes, gt_bboxes, iou_threshold=0.5):
#     # Match predicted boxes to ground truth boxes based on IoU threshold
#     # Returns a list of matched pairs and a list of unmatched predictions
#     pass

# def compute_precision_recall(matches, n_gt):
#     # Calculate precision and recall from the matched boxes
#     # Returns a list of precision and recall values
#     pass

# def compute_ap40(gt_bboxes, pred_bboxes, pred_scores, iou_threshold=0.5):
#     """
#     Compute AP40 for 3D detection.
    
#     Parameters:
#     - gt_bboxes: List of ground truth 3D bounding boxes [x, y, z, l, w, h, heading]
#     - pred_bboxes: List of predicted 3D bounding boxes [x, y, z, l, w, h, heading]
#     - pred_scores: List of confidence scores corresponding to pred_bboxes
#     - iou_threshold: IoU threshold to match predicted boxes with ground truth
    
#     Returns:
#     - ap40: Average Precision at 40 recall levels
#     """
#     n_gt = len(gt_bboxes)
    
#     # Sort predictions by confidence score (descending)
#     sorted_indices = np.argsort(-np.array(pred_scores))
#     pred_bboxes = [pred_bboxes[i] for i in sorted_indices]
#     pred_scores = [pred_scores[i] for i in sorted_indices]
    
#     # Match predicted boxes to ground truth boxes
#     matches = match_boxes(pred_bboxes, gt_bboxes, iou_threshold)
    
#     # Calculate precision and recall at 40 recall levels
#     precision_recall = compute_precision_recall(matches, n_gt)
    
#     # Interpolate precision at 40 evenly spaced recall levels
#     recall_levels = np.linspace(0, 1, 40)
#     precisions = []
    
#     for r in recall_levels:
#         precisions.append(max(p for (p, rc) in precision_recall if rc >= r))
    
#     # Compute AP40 as the average precision across recall levels
#     ap40 = np.mean(precisions)
    
#     return ap40



'''
######################################### REFERENCE DATA ##########################################
'''

# 08/26 14:29:05 - mmengine - INFO - Epoch(val)  [2][  50/1843]    eta: 0:01:45  time: 0.0588  data_time: 0.0022  memory: 4022  
# 08/26 14:29:08 - mmengine - INFO - Epoch(val)  [2][ 100/1843]    eta: 0:01:42  time: 0.0589  data_time: 0.0012  memory: 532  
# 08/26 14:29:11 - mmengine - INFO - Epoch(val)  [2][ 150/1843]    eta: 0:01:39  time: 0.0584  data_time: 0.0011  memory: 554  
# 08/26 14:29:14 - mmengine - INFO - Epoch(val)  [2][ 200/1843]    eta: 0:01:36  time: 0.0582  data_time: 0.0012  memory: 521  
# 08/26 14:29:16 - mmengine - INFO - Epoch(val)  [2][ 250/1843]    eta: 0:01:32  time: 0.0575  data_time: 0.0011  memory: 566  
# 08/26 14:29:19 - mmengine - INFO - Epoch(val)  [2][ 300/1843]    eta: 0:01:30  time: 0.0585  data_time: 0.0012  memory: 531  
# 08/26 14:29:22 - mmengine - INFO - Epoch(val)  [2][ 350/1843]    eta: 0:01:26  time: 0.0571  data_time: 0.0011  memory: 536  
# 08/26 14:29:25 - mmengine - INFO - Epoch(val)  [2][ 400/1843]    eta: 0:01:24  time: 0.0597  data_time: 0.0011  memory: 552  
# 08/26 14:29:28 - mmengine - INFO - Epoch(val)  [2][ 450/1843]    eta: 0:01:21  time: 0.0577  data_time: 0.0011  memory: 544  
# 08/26 14:29:31 - mmengine - INFO - Epoch(val)  [2][ 500/1843]    eta: 0:01:18  time: 0.0600  data_time: 0.0011  memory: 546  
# 08/26 14:29:34 - mmengine - INFO - Epoch(val)  [2][ 550/1843]    eta: 0:01:15  time: 0.0589  data_time: 0.0011  memory: 522  
# 08/26 14:29:37 - mmengine - INFO - Epoch(val)  [2][ 600/1843]    eta: 0:01:12  time: 0.0578  data_time: 0.0012  memory: 482  
# 08/26 14:29:40 - mmengine - INFO - Epoch(val)  [2][ 650/1843]    eta: 0:01:09  time: 0.0565  data_time: 0.0014  memory: 529  
# 08/26 14:29:43 - mmengine - INFO - Epoch(val)  [2][ 700/1843]    eta: 0:01:06  time: 0.0590  data_time: 0.0013  memory: 567  
# 08/26 14:29:46 - mmengine - INFO - Epoch(val)  [2][ 750/1843]    eta: 0:01:03  time: 0.0577  data_time: 0.0012  memory: 514  
# 08/26 14:29:48 - mmengine - INFO - Epoch(val)  [2][ 800/1843]    eta: 0:01:00  time: 0.0562  data_time: 0.0012  memory: 533  
# 08/26 14:29:51 - mmengine - INFO - Epoch(val)  [2][ 850/1843]    eta: 0:00:57  time: 0.0604  data_time: 0.0036  memory: 550  
# 08/26 14:29:54 - mmengine - INFO - Epoch(val)  [2][ 900/1843]    eta: 0:00:55  time: 0.0592  data_time: 0.0014  memory: 510  
# 08/26 14:29:57 - mmengine - INFO - Epoch(val)  [2][ 950/1843]    eta: 0:00:52  time: 0.0590  data_time: 0.0014  memory: 569  
# 08/26 14:30:00 - mmengine - INFO - Epoch(val)  [2][1000/1843]    eta: 0:00:49  time: 0.0576  data_time: 0.0013  memory: 536  
# 08/26 14:30:03 - mmengine - INFO - Epoch(val)  [2][1050/1843]    eta: 0:00:46  time: 0.0570  data_time: 0.0013  memory: 511  
# 08/26 14:30:06 - mmengine - INFO - Epoch(val)  [2][1100/1843]    eta: 0:00:43  time: 0.0568  data_time: 0.0013  memory: 554  
# 08/26 14:30:09 - mmengine - INFO - Epoch(val)  [2][1150/1843]    eta: 0:00:40  time: 0.0583  data_time: 0.0013  memory: 549  
# 08/26 14:30:12 - mmengine - INFO - Epoch(val)  [2][1200/1843]    eta: 0:00:37  time: 0.0576  data_time: 0.0012  memory: 564  
# 08/26 14:30:15 - mmengine - INFO - Epoch(val)  [2][1250/1843]    eta: 0:00:34  time: 0.0583  data_time: 0.0013  memory: 545  
# 08/26 14:30:18 - mmengine - INFO - Epoch(val)  [2][1300/1843]    eta: 0:00:31  time: 0.0581  data_time: 0.0013  memory: 530  
# 08/26 14:30:20 - mmengine - INFO - Epoch(val)  [2][1350/1843]    eta: 0:00:28  time: 0.0576  data_time: 0.0013  memory: 493  
# 08/26 14:30:23 - mmengine - INFO - Epoch(val)  [2][1400/1843]    eta: 0:00:25  time: 0.0569  data_time: 0.0010  memory: 566  
# 08/26 14:30:26 - mmengine - INFO - Epoch(val)  [2][1450/1843]    eta: 0:00:22  time: 0.0586  data_time: 0.0011  memory: 552  
# 08/26 14:30:29 - mmengine - INFO - Epoch(val)  [2][1500/1843]    eta: 0:00:19  time: 0.0565  data_time: 0.0012  memory: 551  
# 08/26 14:30:32 - mmengine - INFO - Epoch(val)  [2][1550/1843]    eta: 0:00:17  time: 0.0580  data_time: 0.0012  memory: 564  
# 08/26 14:30:35 - mmengine - INFO - Epoch(val)  [2][1600/1843]    eta: 0:00:14  time: 0.0589  data_time: 0.0011  memory: 503  
# 08/26 14:30:38 - mmengine - INFO - Epoch(val)  [2][1650/1843]    eta: 0:00:11  time: 0.0572  data_time: 0.0010  memory: 487  
# 08/26 14:30:41 - mmengine - INFO - Epoch(val)  [2][1700/1843]    eta: 0:00:08  time: 0.0565  data_time: 0.0011  memory: 543  
# 08/26 14:30:43 - mmengine - INFO - Epoch(val)  [2][1750/1843]    eta: 0:00:05  time: 0.0577  data_time: 0.0011  memory: 543  
# 08/26 14:30:46 - mmengine - INFO - Epoch(val)  [2][1800/1843]    eta: 0:00:02  time: 0.0572  data_time: 0.0011  memory: 547  



# Converting 3D prediction to KITTI format
# [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 1843/1843, 473.1 task/s, elapsed: 4s, ETA:     0s
# Result is saved to /tmp/tmputpmpblk/results/pred_instances_3d.pkl.



# 08/26 14:31:01 - mmengine - INFO - Results of pred_instances_3d:

# ----------- AP11 Results ------------

# Car AP11@0.70, 0.70, 0.70:
# bbox AP11:78.4568, 63.9296, 59.2733
# bev  AP11:78.4496, 68.4197, 64.3918
# 3d   AP11:35.3031, 30.0891, 29.1243
# aos  AP11:76.86, 61.96, 56.98
# Car AP11@0.70, 0.50, 0.50:
# bbox AP11:78.4568, 63.9296, 59.2733
# bev  AP11:88.2605, 83.8585, 77.3190
# 3d   AP11:87.7581, 80.3100, 75.2511
# aos  AP11:76.86, 61.96, 56.98

# ----------- AP40 Results ------------

# Car AP40@0.70, 0.70, 0.70:
# bbox AP40:79.2700, 65.0151, 58.6590
# bev  AP40:79.8558, 68.9387, 64.3797
# 3d   AP40:31.1601, 26.1343, 23.8225
# aos  AP40:77.59, 62.73, 56.19
# Car AP40@0.70, 0.50, 0.50:
# bbox AP40:79.2700, 65.0151, 58.6590
# bev  AP40:92.4878, 85.0034, 80.4381
# 3d   AP40:91.6861, 81.3147, 75.3186
# aos  AP40:77.59, 62.73, 56.19






'''
######################################### ACTUAL FUNCTION ##########################################
'''

# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)


@METRICS.register_module()
class MinervaMetric(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """






    # TODO: Sistema questa inizializzazione per togliere tutta la roba inutile
    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Kitti metric'
        super(MinervaMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')






    # TODO: Do something (erase or analyze)
    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        """Convert loading annotations to Kitti annotations.

        Args:
            data_infos (dict): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of Kitti annotations.
        """
        data_annos = data_infos['data_list']
        if not self.format_only:
            cat2label = data_infos['metainfo']['categories']
            label2cat = dict((v, k) for (k, v) in cat2label.items())
            assert 'instances' in data_annos[0]
            for i, annos in enumerate(data_annos):
                if len(annos['instances']) == 0:
                    kitti_annos = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    }
                else:
                    kitti_annos = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'location': [],
                        'dimensions': [],
                        'rotation_y': [],
                        'score': []
                    }
                    for instance in annos['instances']:
                        label = instance['bbox_label']
                        kitti_annos['name'].append(label2cat[label])
                        # kitti_annos['truncated'].append(instance['truncated'])
                        # kitti_annos['occluded'].append(instance['occluded'])
                        # kitti_annos['alpha'].append(instance['alpha'])
                        # kitti_annos['bbox'].append(instance['bbox'])
                        kitti_annos['location'].append(instance['bbox_3d'][:3])
                        kitti_annos['dimensions'].append(
                            instance['bbox_3d'][3:6])
                        kitti_annos['rotation_y'].append(
                            instance['bbox_3d'][6])
                        # kitti_annos['score'].append(instance['score'])
                    for name in kitti_annos:
                        kitti_annos[name] = np.array(kitti_annos[name])
                data_annos[i]['kitti_annos'] = kitti_annos
        return data_annos






    # Description of the function.
    #       - Is called batch by batch (in our case just 1 scan, see "minerva_only_lidar_dataset.py"), 
    #         by the file called "mmengine/evaluator/evaluator.py"
    #       - Receives a list of dictionaries (always of length 1, independently from the number of
    #         instances) called "data_samples"
    #       - Each dictionary contains the information about the predictions, including bboxes and
    #         scores and labels (a.k.a. number corresponding to the category) etc.
    #       - The dictionary is then "transformed" in another list of dictionaries (again, always of 
    #         length 1) called "self.results"
    #       - The only fields that are saved are the ones regarding 
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        
        for data_sample in data_samples:
            result = dict()
            
            # The 3D predictions (bboxes) are saved and transferred to the 'cpu' (from the 'gpu')
            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            
            # Part below is commented because is not used (we don't do 2D-predictions)
            # pred_2d = data_sample['pred_instances']
            # for attr_name in pred_2d:
            #     pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            # result['pred_instances'] = pred_2d
            
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            
            self.results.append(result)






    # Description of the function.
    #  
    #       - Is called only at the end of the processing, for all the batches, by the file
    #         called "mmengine/evaluator/metric.py"
    #  
    #       - The whole flow of data is:    - "mmengine/evaluator/metric.py"
    #                                       - "mmengine/evaluator/evaluator.py"
    #                                       - "mmengine/runner/loops.py" (around line 382)
    #                                       - The data "metrics" are then passed to a hook 
    #                                         of type "LoggerHook" that lauches the function 
    #                                         "after_val_epoch"
    #                                       - Such function has as the parameters the one 
    #                                         "Optional[Dict[str, float]] = None"
    #                                       - ...so this is the final format, nothing more 
    #                                         and nothing else
    #  
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)

        self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)
        
        
        



        print(self.results)        
        # for i, element in enumerate(self.results):
        #     if element[]
        
        print(self.data_infos)




        return {
            "porcoddio": 10.,
            "e la madonna": .5,
            "quella puttana": 0.,
            "maledetta": 1.8
        }
        
        



        
        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.submission_prefix)}')
            return metric_dict

        gt_annos = [
            self.data_infos[result['sample_idx']]['kitti_annos']
            for result in results
        ]

        for metric in self.metrics:
            ap_dict = self.kitti_evaluate(
                result_dict,
                gt_annos,
                metric=metric,
                logger=logger,
                classes=self.classes)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    
    
    
    
    
    ##  
    #  
    def kitti_evaluate(self,
                       results_dict: dict,
                       gt_annos: List[dict],
                       metric: Optional[str] = None,
                       classes: Optional[List[str]] = None,
                       logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in KITTI protocol.

        Args:
            results_dict (dict): Formatted results of the dataset.
            gt_annos (List[dict]): Contain gt information of each sample.
            metric (str, optional): Metrics to be evaluated. Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        ap_dict = dict()
        for name in results_dict:
            if name == 'pred_instances' or metric == 'img_bbox':
                eval_types = ['bbox']
            else:
                eval_types = ['bbox', 'bev', '3d']
            ap_result_str, ap_dict_ = kitti_eval(
                gt_annos, results_dict[name], classes, eval_types=eval_types)
            for ap_type, ap in ap_dict_.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')

            print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)

        return ap_dict

    def format_results(
        self,
        results: List[dict],
        pklfile_prefix: Optional[str] = None,
        submission_prefix: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to pkl file.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing the
            formatted result, tmp_dir is the temporal directory created for
            saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]
        for name in results[0]:
            if submission_prefix is not None:
                submission_prefix_ = osp.join(submission_prefix, name)
            else:
                submission_prefix_ = None
            if pklfile_prefix is not None:
                pklfile_prefix_ = osp.join(pklfile_prefix, name) + '.pkl'
            else:
                pklfile_prefix_ = None
            if 'pred_instances' in name and '3d' in name and name[
                    0] != '_' and results[0][name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti(net_outputs,
                                                      sample_idx_list, classes,
                                                      pklfile_prefix_,
                                                      submission_prefix_)
                result_dict[name] = result_list_
            elif name == 'pred_instances' and name[0] != '_' and results[0][
                    name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti2d(
                    net_outputs, sample_idx_list, classes, pklfile_prefix_,
                    submission_prefix_)
                result_dict[name] = result_list_
        return result_dict, tmp_dir

    def bbox2result_kitti(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting 3D prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            info = self.data_infos[sample_idx]
            # Here default used 'CAM2' to compute metric. If you want to
            # use another camera, please modify it.
            # image_shape = (info['images'][self.default_cam_key]['height'],
            #                info['images'][self.default_cam_key]['width'])
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                pred_box_type_3d = box_dict['pred_box_type_3d']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    # bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[2:] = bbox[2:]
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    if pred_box_type_3d == CameraInstance3DBoxes:
                        anno['alpha'].append(-np.arctan2(box[0], box[2]) +
                                             box[6])
                    elif pred_box_type_3d == LiDARInstance3DBoxes:
                        anno['alpha'].append(
                            -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            anno['sample_idx'] = np.array(
                [sample_idx] * len(anno['score']), dtype=np.int64)

            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def bbox2result_kitti2d(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting 2D prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = sample_idx_list[i]

            num_example = 0
            bbox = bboxes_per_sample['bboxes']
            for i in range(bbox.shape[0]):
                anno['name'].append(class_names[int(
                    bboxes_per_sample['labels'][i])])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0.0)
                anno['bbox'].append(bbox[i, :4])
                # set dimensions (height, width, length) to zero
                anno['dimensions'].append(
                    np.zeros(shape=[3], dtype=np.float32))
                # set the 3D translation to (-1000, -1000, -1000)
                anno['location'].append(
                    np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                anno['rotation_y'].append(0.0)
                anno['score'].append(bboxes_per_sample['scores'][i])
                num_example += 1

            if num_example == 0:
                anno = dict(
                    name=np.array([]),
                    truncated=np.array([]),
                    occluded=np.array([]),
                    alpha=np.array([]),
                    bbox=np.zeros([0, 4]),
                    dimensions=np.zeros([0, 3]),
                    location=np.zeros([0, 3]),
                    rotation_y=np.array([]),
                    score=np.array([]),
                )
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}

            anno['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        if submission_prefix is not None:
            # save file in submission format
            mmengine.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = sample_idx_list[i]
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict: dict, info: dict) -> dict:
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (Tensor): Scores of boxes.
                - labels_3d (Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

            - bbox (np.ndarray): 2D bounding boxes.
            - box3d_camera (np.ndarray): 3D bounding boxes in
              camera coordinate.
            - box3d_lidar (np.ndarray): 3D bounding boxes in
              LiDAR coordinate.
            - scores (np.ndarray): Scores of boxes.
            - label_preds (np.ndarray): Class label predictions.
            - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        # Here default used 'CAM2' to compute metric. If you want to
        # use another camera, please modify it.
        # lidar2cam = np.array(
        #     info['images'][self.default_cam_key]['lidar2cam']).astype(
        #         np.float32)
        P2 = np.array(info['images'][self.default_cam_key]['cam2img']).astype(
            np.float32)
        # img_shape = (info['images'][self.default_cam_key]['height'],
        #              info['images'][self.default_cam_key]['width'])
        P2 = box_preds.tensor.new_tensor(P2)

        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_camera = box_preds.convert_to(Box3DMode.CAM)
            box_preds_lidar = box_preds
        elif isinstance(box_preds, CameraInstance3DBoxes):
            box_preds_camera = box_preds
            box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
                                                   np.linalg.inv(lidar2cam))

        box_corners = box_preds_camera.corners
        # box_corners_in_image = points_cam2img(box_corners, P2)
        # # box_corners_in_image: [N, 8, 2]
        # minxy = torch.min(box_corners_in_image, dim=1)[0]
        # maxxy = torch.max(box_corners_in_image, dim=1)[0]
        # box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        # image_shape = box_preds.tensor.new_tensor(img_shape)
        # valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
        #                   (box_2d_preds[:, 1] < image_shape[0]) &
        #                   (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds_lidar
        if isinstance(box_preds, LiDARInstance3DBoxes):
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds_lidar.center > limit_range[:3]) &
                              (box_preds_lidar.center < limit_range[3:]))
            # valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
            valid_inds = valid_pcd_inds.all(-1)
        else:
            # valid_inds = valid_cam_inds
            valid_inds = None

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                pred_box_type_3d=type(box_preds),
                box3d_camera=box_preds_camera[valid_inds].numpy(),
                box3d_lidar=box_preds_lidar[valid_inds].numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                pred_box_type_3d=type(box_preds),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)
