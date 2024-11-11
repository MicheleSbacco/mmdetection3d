# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .voxelnet import VoxelNet

# Added imports for the time computation
from demo.json_handler import JSONHandler
import torch
import time

@MODELS.register_module()
class DynamicVoxelNet(VoxelNet):
    r"""VoxelNet using `dynamic voxelization
    <https://arxiv.org/abs/1910.06528>`_.
    """



    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 save_losses_on_file = True,            # Related to VoxelNet (1)
                 losses_file_destination_path = None    # Related to VoxelNet (2)
                 ) -> None:
        super().__init__(
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            save_losses_on_file=save_losses_on_file,                        # Related to VoxelNet (1)
            losses_file_destination_path=losses_file_destination_path)      # Related to VoxelNet (2)



    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features, feature_coors = self.voxel_encoder(
            voxel_dict['voxels'], voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x




    # Copied function from VoxelNet.
    #   - Just need to modify the part related to the passed arguments
    def extract_feat_test(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        
        # Added lines to initialize the handler, for time computation
        out_file = '/home/michele/code/michele_mmdet3d/data/minerva_polimove/inference_times.json'
        handler = JSONHandler(out_file)
        
        voxel_dict = batch_inputs_dict['voxels']
        torch.cuda.synchronize()
        vox_enc0 = time.time()                                              # Added for time computation
        voxel_features, feature_coors = self.voxel_encoder(
            voxel_dict['voxels'],
            voxel_dict['coors'])
        torch.cuda.synchronize()
        vox_enc1 = time.time()                                              # Added for time computation
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        torch.cuda.synchronize()
        mid_enc0 = time.time()                                              # Added for time computation
        x = self.middle_encoder(voxel_features, feature_coors,
                                batch_size)
        torch.cuda.synchronize()
        mid_enc1 = time.time()                                              # Added for time computation
        torch.cuda.synchronize()
        back_bone0 = time.time()                                            # Added for time computation
        x = self.backbone(x)
        torch.cuda.synchronize()
        back_bone1 = time.time()                                            # Added for time computation
        if self.with_neck:
            torch.cuda.synchronize()
            neck0 = time.time()                                             # Added for time computation
            x = self.neck(x)
            torch.cuda.synchronize()
            neck1 = time.time()                                             # Added for time computation
        
        # Added lines to handle the json file
        handler.update_dictionary({'Voxel encoder start': vox_enc0,
                                   'Voxel encoder delta_t': (vox_enc1-vox_enc0),
                                   'Middle encoder delta_t': (mid_enc1-mid_enc0),
                                   'Backbone delta_t': (back_bone1-back_bone0),
                                   'Neck delta_t': (neck1-neck0)})

        return x
