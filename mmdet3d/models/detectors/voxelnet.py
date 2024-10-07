# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector

# Added import for the computation of time
import time
wanna_print = True

@MODELS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        if wanna_print:                                                                                                     # Added for time computation
            out_file = '/home/michele/iac_code/michele_mmdet3d/data/minerva_polimove/inference_times.txt'
            vox_enc = time.time()
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        if wanna_print:                                                                                                     # Added for time computation
            vox_enc1 = time.time()
            # print(f"\tTime for voxel encoder: {vox_enc1-vox_enc}. Start of voxel encoder: {vox_enc}")
            with open(out_file, "a") as file:
                file.write(f"\tTime for voxel encoder: {vox_enc1-vox_enc}. Start of voxel encoder: {vox_enc}\n")
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        if wanna_print:                                                                                                     # Added for time computation
            mid_enc = time.time()
        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        if wanna_print:                                                                                                     # Added for time computation
            mid_enc1 = time.time()
            # print(f"\tTime for middle encoder: {mid_enc1-mid_enc}. Start of middle encoder: {mid_enc}")
            with open(out_file, "a") as file:
                file.write(f"\tTime for middle encoder: {mid_enc1-mid_enc}. Start of middle encoder: {mid_enc}\n")
            back_bone = time.time()
        x = self.backbone(x)
        if wanna_print:                                                                                                     # Added for time computation
            back_bone1 = time.time()
            # print(f"\tTime for backbone: {back_bone1-back_bone}. Start of backbone: {back_bone}")
            with open(out_file, "a") as file:
                file.write(f"\tTime for backbone: {back_bone1-back_bone}. Start of backbone: {back_bone}\n")
        if self.with_neck:
            if wanna_print:                                                                                                 # Added for time computation
                neck = time.time()
            x = self.neck(x)
            if wanna_print:                                                                                                 # Added for time computation
                neck1 = time.time()
                # print(f"\tTime for neck: {neck1-neck}. Start of neck: {neck}")
                with open(out_file, "a") as file:
                    file.write(f"\tTime for neck: {neck1-neck}. Start of neck: {neck}\n")
        return x
