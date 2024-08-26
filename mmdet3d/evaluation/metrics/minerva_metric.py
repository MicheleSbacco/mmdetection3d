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






######################################### GENERAL DESCRIPTION #########################################

# For now: just copied straight-away from Dona's function



######################################### "STANDARD" KITTI EVALUATION #########################################

'''
08/26 14:29:05 - mmengine - INFO - Epoch(val)  [2][  50/1843]    eta: 0:01:45  time: 0.0588  data_time: 0.0022  memory: 4022  
08/26 14:29:08 - mmengine - INFO - Epoch(val)  [2][ 100/1843]    eta: 0:01:42  time: 0.0589  data_time: 0.0012  memory: 532  
08/26 14:29:11 - mmengine - INFO - Epoch(val)  [2][ 150/1843]    eta: 0:01:39  time: 0.0584  data_time: 0.0011  memory: 554  
08/26 14:29:14 - mmengine - INFO - Epoch(val)  [2][ 200/1843]    eta: 0:01:36  time: 0.0582  data_time: 0.0012  memory: 521  
08/26 14:29:16 - mmengine - INFO - Epoch(val)  [2][ 250/1843]    eta: 0:01:32  time: 0.0575  data_time: 0.0011  memory: 566  
08/26 14:29:19 - mmengine - INFO - Epoch(val)  [2][ 300/1843]    eta: 0:01:30  time: 0.0585  data_time: 0.0012  memory: 531  
08/26 14:29:22 - mmengine - INFO - Epoch(val)  [2][ 350/1843]    eta: 0:01:26  time: 0.0571  data_time: 0.0011  memory: 536  
08/26 14:29:25 - mmengine - INFO - Epoch(val)  [2][ 400/1843]    eta: 0:01:24  time: 0.0597  data_time: 0.0011  memory: 552  
08/26 14:29:28 - mmengine - INFO - Epoch(val)  [2][ 450/1843]    eta: 0:01:21  time: 0.0577  data_time: 0.0011  memory: 544  
08/26 14:29:31 - mmengine - INFO - Epoch(val)  [2][ 500/1843]    eta: 0:01:18  time: 0.0600  data_time: 0.0011  memory: 546  
08/26 14:29:34 - mmengine - INFO - Epoch(val)  [2][ 550/1843]    eta: 0:01:15  time: 0.0589  data_time: 0.0011  memory: 522  
08/26 14:29:37 - mmengine - INFO - Epoch(val)  [2][ 600/1843]    eta: 0:01:12  time: 0.0578  data_time: 0.0012  memory: 482  
08/26 14:29:40 - mmengine - INFO - Epoch(val)  [2][ 650/1843]    eta: 0:01:09  time: 0.0565  data_time: 0.0014  memory: 529  
08/26 14:29:43 - mmengine - INFO - Epoch(val)  [2][ 700/1843]    eta: 0:01:06  time: 0.0590  data_time: 0.0013  memory: 567  
08/26 14:29:46 - mmengine - INFO - Epoch(val)  [2][ 750/1843]    eta: 0:01:03  time: 0.0577  data_time: 0.0012  memory: 514  
08/26 14:29:48 - mmengine - INFO - Epoch(val)  [2][ 800/1843]    eta: 0:01:00  time: 0.0562  data_time: 0.0012  memory: 533  
08/26 14:29:51 - mmengine - INFO - Epoch(val)  [2][ 850/1843]    eta: 0:00:57  time: 0.0604  data_time: 0.0036  memory: 550  
08/26 14:29:54 - mmengine - INFO - Epoch(val)  [2][ 900/1843]    eta: 0:00:55  time: 0.0592  data_time: 0.0014  memory: 510  
08/26 14:29:57 - mmengine - INFO - Epoch(val)  [2][ 950/1843]    eta: 0:00:52  time: 0.0590  data_time: 0.0014  memory: 569  
08/26 14:30:00 - mmengine - INFO - Epoch(val)  [2][1000/1843]    eta: 0:00:49  time: 0.0576  data_time: 0.0013  memory: 536  
08/26 14:30:03 - mmengine - INFO - Epoch(val)  [2][1050/1843]    eta: 0:00:46  time: 0.0570  data_time: 0.0013  memory: 511  
08/26 14:30:06 - mmengine - INFO - Epoch(val)  [2][1100/1843]    eta: 0:00:43  time: 0.0568  data_time: 0.0013  memory: 554  
08/26 14:30:09 - mmengine - INFO - Epoch(val)  [2][1150/1843]    eta: 0:00:40  time: 0.0583  data_time: 0.0013  memory: 549  
08/26 14:30:12 - mmengine - INFO - Epoch(val)  [2][1200/1843]    eta: 0:00:37  time: 0.0576  data_time: 0.0012  memory: 564  
08/26 14:30:15 - mmengine - INFO - Epoch(val)  [2][1250/1843]    eta: 0:00:34  time: 0.0583  data_time: 0.0013  memory: 545  
08/26 14:30:18 - mmengine - INFO - Epoch(val)  [2][1300/1843]    eta: 0:00:31  time: 0.0581  data_time: 0.0013  memory: 530  
08/26 14:30:20 - mmengine - INFO - Epoch(val)  [2][1350/1843]    eta: 0:00:28  time: 0.0576  data_time: 0.0013  memory: 493  
08/26 14:30:23 - mmengine - INFO - Epoch(val)  [2][1400/1843]    eta: 0:00:25  time: 0.0569  data_time: 0.0010  memory: 566  
08/26 14:30:26 - mmengine - INFO - Epoch(val)  [2][1450/1843]    eta: 0:00:22  time: 0.0586  data_time: 0.0011  memory: 552  
08/26 14:30:29 - mmengine - INFO - Epoch(val)  [2][1500/1843]    eta: 0:00:19  time: 0.0565  data_time: 0.0012  memory: 551  
08/26 14:30:32 - mmengine - INFO - Epoch(val)  [2][1550/1843]    eta: 0:00:17  time: 0.0580  data_time: 0.0012  memory: 564  
08/26 14:30:35 - mmengine - INFO - Epoch(val)  [2][1600/1843]    eta: 0:00:14  time: 0.0589  data_time: 0.0011  memory: 503  
08/26 14:30:38 - mmengine - INFO - Epoch(val)  [2][1650/1843]    eta: 0:00:11  time: 0.0572  data_time: 0.0010  memory: 487  
08/26 14:30:41 - mmengine - INFO - Epoch(val)  [2][1700/1843]    eta: 0:00:08  time: 0.0565  data_time: 0.0011  memory: 543  
08/26 14:30:43 - mmengine - INFO - Epoch(val)  [2][1750/1843]    eta: 0:00:05  time: 0.0577  data_time: 0.0011  memory: 543  
08/26 14:30:46 - mmengine - INFO - Epoch(val)  [2][1800/1843]    eta: 0:00:02  time: 0.0572  data_time: 0.0011  memory: 547  



Converting 3D prediction to KITTI format
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 1843/1843, 473.1 task/s, elapsed: 4s, ETA:     0s
Result is saved to /tmp/tmputpmpblk/results/pred_instances_3d.pkl.



08/26 14:31:01 - mmengine - INFO - Results of pred_instances_3d:

----------- AP11 Results ------------

Car AP11@0.70, 0.70, 0.70:
bbox AP11:78.4568, 63.9296, 59.2733
bev  AP11:78.4496, 68.4197, 64.3918
3d   AP11:35.3031, 30.0891, 29.1243
aos  AP11:76.86, 61.96, 56.98
Car AP11@0.70, 0.50, 0.50:
bbox AP11:78.4568, 63.9296, 59.2733
bev  AP11:88.2605, 83.8585, 77.3190
3d   AP11:87.7581, 80.3100, 75.2511
aos  AP11:76.86, 61.96, 56.98

----------- AP40 Results ------------

Car AP40@0.70, 0.70, 0.70:
bbox AP40:79.2700, 65.0151, 58.6590
bev  AP40:79.8558, 68.9387, 64.3797
3d   AP40:31.1601, 26.1343, 23.8225
aos  AP40:77.59, 62.73, 56.19
Car AP40@0.70, 0.50, 0.50:
bbox AP40:79.2700, 65.0151, 58.6590
bev  AP40:92.4878, 85.0034, 80.4381
3d   AP40:91.6861, 81.3147, 75.3186
aos  AP40:77.59, 62.73, 56.19
'''






@METRICS.register_module()
class MinervaMetric(BaseMetric):
    """Losses evaluation metric.
    Args:
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """



    def __init__(self,
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = ''
        super(MinervaMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.backend_args = backend_args



    # This function is called by "mmengine/evaluator/evaluator.py"
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        print("\n\n%s\n\n%s\n\n%s\n\n" %(data_samples, type(data_samples), len(data_samples)))
        self.results.append(data_samples[1])



    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.
        Args:
            results (List[dict]): The processed results of the whole dataset.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        logger.info(
                f'======================================================================== VALIDATION RESULTS ========================================================================')
        sum_losses = {'loss': 0,'loss_cls': 0, 'loss_bbox': 0, 'loss_dir': 0}

        # Iterate over each element in the list and accumulate the values
        for item in results:
            for key, values in item.items():
                sum_losses[key] += values[0].item()
                sum_losses["loss"] += values[0].item()
        # Calculate the average for each loss type
        metric_dict = {key: value / len(results) for key, value in sum_losses.items()}

        return metric_dict