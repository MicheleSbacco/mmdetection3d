# For reference:        - go to file "kitty_dataset.py" in this same folder
#                       - go to page "Docs" --> "Advanced Guides" --> "Customization" --> "Customize Datasets"
#
# The page "Docs" --> "User Guides" --> "Train and Test" --> "Train with Customized Datasets" does not provide many
# details, just a brief introduction
#
# What is the use of this file? It is a class that is dictionary-called when creating the dataset in the folder 
# "configs/_base_/datasets/michele_custom_dataset.py"



## Imports suggested by the Docs
import mmengine
from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset
## Added imports
import numpy as np



@DATASETS.register_module()
class MicheleCustomDataset(Det3DDataset):

    # Docs: replace with all the classes in customized pkl info file
    # 
    # Apparently, here need to add all the classes that I added in the ".pkl" file. Then I will be able to remove them later
    # in the code.
    # "kitti_dataset.py" also adds the key "palette" which is a tuple containing the RGB colors for each class
    # 
    # I will just add all the classes of the KITTI dataset, without the palette key
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc')
    }

    def parse_ann_info(self, info):
        """
        Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info