'''
######################################### GENERAL DESCRIPTION ##########################################
'''

# Tried with Dona's version but it's a mess between various files. Will try to make it from scratch to make it lighter
#
#  
# New idea is to use the predictions and manually compute the AP40 with the use o the functions already implemented by
# mmdet3d in the file "mmdet3d/structures/ops/iou3dcalculator.py". 
# The metric would be the 3D-metric (not bbox because it projects on the image plane, does not make sense without any
# image).
#
#  
# Recap of the analysis for the printing: 
#       - the function "compute_metrics" must return a dictionary of "key[str]: value[float]" but it is printed in the 
#         "ugly" way (like the first block below here)
#       - the function "kitti_evaluate" has been used as a reference for the "beautiful" print, with the specific line 
#         being "print_log(f'Results of " that can be directly integrated inside "compute_metrics"
#
#  
# TODO: Remember to check if it is possible to add the "loss" value in loops.py
# TODO: See if it is possible to add other metrics apart from the 3d metric 
# TODO: Actually the AP40 must be computed for different values of the iou_threshold






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



# Added import for the computation of the IoU
from mmdet3d.structures.ops.iou3d_calculator import BboxOverlaps3D



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
        self.default_prefix = 'Minerva'
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
        
        # Initialize the variable that will contain the bboxes for the evaluation of the AP40
        self.bboxes = []
        # Initialize the iou threshold for the computation of true positives etc. (according to
        # the iou) TODO: make it settable in the initialization
        self.iou_threshold = 0.5
        # Initialize the type of bbox 
        self.box_type = 'lidar'






    # This function:
    #   - returns a list of dictionaries
    #   - each dictionary of the list corresponds to a "scan" of the LiDAR
    #   - each dictionary contains:
    #       - the sample index (in timestamps)
    #       - a tensor of dimensions (N, 7) where 7 are the dimentions of the bbox, and 
    #         N the number of bboxes in the single scan
    def convert_pkl_to_gt_bboxes(self, pkl_infos):
        
        # Create the list that will contain the dictionaries
        dictionary_list = []
        
        # Cycle through the number of scans
        for scan in pkl_infos['data_list']:
            # Create the local dictionary, to be appended to the list later on
            local_dictionary = {
                'sample_index' : scan['sample_idx']
            }
            # Temporary variable where to create the tensor
            local_tensor = torch.zeros(len(scan['instances']), 7)
            
            # Cycle through the bboxes in the same scan
            for i, instance in enumerate(scan['instances']):
                # Assign the row of the tensor
                local_tensor[i] = torch.tensor(instance['bbox_3d'])
            
            # Assign the new field to the local dictionary
            local_dictionary['bboxes'] = local_tensor
            # Update the list with the new tensor
            dictionary_list.append(local_dictionary)

        return dictionary_list






    # Description of the function.
    #       - Is called batch by batch (in our case just 1 scan, see "minerva_only_lidar_dataset.py"), 
    #         by the file called "mmengine/evaluator/evaluator.py"
    #       - Receives a list of dictionaries (always of length 1, independently from the number of
    #         instances) called "data_samples"
    #       - Each dictionary contains the information about the predictions, including bboxes and
    #         scores and labels (a.k.a. number corresponding to the category) etc.
    #       - The dictionary is then "transformed" in another list of dictionaries (again, always of 
    #         length 1) called "self.results"
    #       - Only some selected fields are saved, not all of them
    # 
    # 
    # UPDATE:
    #       - A whole new section has been added to process the gt_bboxes in order to compute the 
    #         metrics
    #       - In this way there is no need to do strange computations in several function
    #       - The dictionary includes:  - sample_idx (starting from 0)
    #                                   - timestamp 
    #                                   - pred_bboxes (torch.tensor) dimension (N, 7)
    #                                   - pred_scores (torch.tensor) dimension (N)
    #                                   - gt_bboxes (torch.tensor) dimension (N, 7)
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        
        ##################################### OLD PART #####################################
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

        ##################################### NEW PART #####################################
        # Just process the first element (not found cases when there are more than just one element)
        data_sample = data_samples[0]
        # Create the dictionary and update all of its values
        new_dictionary = {}
        new_dictionary['sample_idx'] = data_sample['sample_idx']
        new_dictionary['timestamp'] = str(data_sample['lidar_path']).split('/')[-1].split('.bin')[0]
        new_dictionary['pred_bboxes'] = data_sample['pred_instances_3d']['bboxes_3d'].tensor
        new_dictionary['pred_scores'] = data_sample['pred_instances_3d']['scores_3d']
        # If has been added to deal with Riccardo's pipeline
        if data_sample['eval_ann_info'] is not None:
            new_dictionary['gt_bboxes'] = data_sample['eval_ann_info']['gt_bboxes_3d'].tensor
        else: 
            new_dictionary['gt_bboxes'] = data_sample['gt_instances_3d']['bboxes_3d'].tensor
        # Append the dictionary to the list of dictionaries
        self.bboxes.append(new_dictionary)






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
    #                                         and nothing less
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

        # ATTENTION:
        #   - The part below here is not used anymore, but the functions have been left
        #   - Everything is now done inside the "self.process()" function

        # # Download the ground truth information from the ".pkl" file
        # pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        # # Elaborate the data in order to save a list of dictionaries
        # self.gt_bboxes = self.convert_pkl_to_gt_bboxes(pkl_infos)
    
        # Create empty lists for precision and recall
        precisions = []
        recalls = []
        
        # Cycle through the dictionaries (one for each scan in the dataset)
        for dict in self.bboxes:
            # The bboxes come already sorted, so there's no need to sort them again
            pass
            # Compute the precision and recall for the scan
            precision, recall = self.compute_precision_recall(dict['pred_bboxes'], dict['gt_bboxes'])
            precisions.append(precision)
            recalls.append(recall)

        # Aggregate all precision-recall curves into a single one
        precisions = np.array(sorted(precisions, reverse=True))
        recalls = np.array(sorted(recalls))

        # Compute AP40
        ap40 = self.compute_ap40(precisions, recalls)
        
        # Prepare the "cool print"
        pre_print = "\n\n---------------------------------------------------------\nResults for validation:\n\n" + \
                    "\t(Method)\t(Metric)\t(Threshold)\t(Value)\n"
        post_print = "\n---------------------------------------------------------\n"
        print_log(f'{pre_print}\tAP40\t\t3D metric\t@ IoU=0.5:\t{ap40}{post_print}', logger=logger)

        # Return the dictionary with "metric: value" for the "ugly" print
        return {'AP40(3d metric)': ap40}






    # This function, adapted from ChatGPT, is used to compute the precision and recall of the single LiDAR scan
    # TODO: understand what it does
    def compute_precision_recall(self, pred_bboxes, gt_bboxes):
        # Initialize the instance of BboxOverlaps3D to compute the IoU
        iou_computer = BboxOverlaps3D(self.box_type)

        # Initialize the number of true positives and false positives
        tp = 0
        fp = 0
        # Create a set (NO repetition!!) that tracks the indeces of matched ground truths
        matched_gts = set()

        # Many steps:
        #       - cycle through the predictions
        #       - then through the ground truths, to find the best match for the prediction
        #       - when (and if) best match has been found:
        #           - update the tp if over the iou_threshold and not already matched
        #           - otherwise update the fp
        for i in range(pred_bboxes.size(0)):
            # Initialize values
            best_iou = 0
            best_gt_idx = -1
            # Internal for loop
            for j in range(gt_bboxes.size(0)):
                # Create the two tensors in such a way that BboxOverlaps3D likes it
                tensor1 = torch.Tensor(1, 7)
                tensor1[0] = pred_bboxes[i]
                tensor2 = torch.Tensor(1, 7)
                tensor2[0] = gt_bboxes[j]
                # Compute the IoU
                iou = iou_computer(tensor1, tensor2)
                # Check if the match has to be updated
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            # Update tp or fp
            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gts:
                tp += 1
                matched_gts.add(best_gt_idx)
            else:
                fp += 1

        # False negatives are ground truths that were not matched
        fn = gt_bboxes.size(0) - len(matched_gts)

        # Calculate precision and recall, and return them
        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        return precision, recall






    # This function computes the ap40, depending on the precisions and recalls
    def compute_ap40(self, precisions, recalls):
        
        recall_levels = np.linspace(0, 1, 40)
        ap40 = 0.0

        for recall_level in recall_levels:
            # Find the highest precision for the recall level or below
            precisions_at_recall = precisions[recalls >= recall_level]
            if precisions_at_recall.size > 0:
                ap40 += np.max(precisions_at_recall)

        ap40 /= len(recall_levels)
        return ap40