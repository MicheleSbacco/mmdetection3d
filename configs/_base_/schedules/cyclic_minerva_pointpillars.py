'''
######################################### GENERAL DESCRIPTION, PARAMETERS ##########################################
'''
# This file was created with a merge between "cyclic-40e" (base) and Kitti+Pointpillars (modifications and updates).

# Comment on the lr by authors:     The learning rate set in the cyclic schedule is the initial learning rate  
#                                   rather than the max learning rate.
#                                   How the learning rate changes is defined by the parameter scheduler
lr = 0.0001
epoch_num = 30
validation_interval = 3
# Comment on the max_norm by authors:   max_norm=35 is slightly better than 10 for PointPillars in the earlier
#                                       development of the codebase thus we keep the setting. But we do not
#                                       specifically tune this parameter.
max_norm = 5






'''
######################################### OPTIMIZER ##########################################
'''

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=lr, 
        betas=(0.95, 0.99), 
        weight_decay=0.01),
    clip_grad=dict(
        max_norm=max_norm, 
        norm_type=2))






'''
######################################### PARAMETER SCHEDULER ##########################################
'''

param_scheduler = [

###################################### LEARNING RATE
                                                                ########### KITTY STYLE
    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=epoch_num,
    #     eta_min=lr * 10,
    #     begin=0,
    #     end=epoch_num * 0.4,
    #     by_epoch=True,
    #     convert_to_iter_based=True),
    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=epoch_num,
    #     eta_min=lr * 1e-3,
    #     begin=epoch_num * 0.4,
    #     end=epoch_num,
    #     by_epoch=True,
    #     convert_to_iter_based=True),
                                                                ########### CONSTANT, HALF L.R. AT HALF PATH
    # dict(
    #     type = 'ConstantLR',
    #     factor = 1,
    #     begin = 0,
    #     end = epoch_num*0.5),
    # dict(
    #     type = 'ConstantLR',
    #     factor = 0.5,
    #     begin = epoch_num*0.5,
    #     end = epoch_num),
                                                                ########### ALWAYS CONSTANT
    dict(
        type = 'ConstantLR',
        factor = 1,
        begin = 0,
        end = epoch_num),
                                                                ########### CONSTANT, INCREASING OF x10 FACTOR
    # dict(
    #     type = 'ConstantLR',
    #     factor = 0.1,
    #     begin = 0,
    #     end = epoch_num*0.5),
    # dict(
    #     type = 'ConstantLR',
    #     factor = 1,
    #     begin = epoch_num*0.5,
    #     end = epoch_num),

###################################### MOMENTUM
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]






'''
######################################### CONFIGURATIONS ##########################################
'''

# Remember that the actual number of epochs is epoch_num*N where N is the parameter of
#   'RepeatDataset', times=N 
# in the dataset file
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=validation_interval)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs)----------------->(num_workers in _base_/dataset)
#                           x 
#                         (4 samples per GPU)------>(batch_size in _base_/dataset)
auto_scale_lr = dict(enable=False, base_batch_size=50)