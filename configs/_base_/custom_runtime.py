default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # Custom visualization hook, to be able to visualize the validation results
    visualization=dict(
        type='Det3DVisualizationHook',
        draw=True,
        interval=1,
        score_thr = 0.1,
        show=True,
        vis_task='lidar_det',
        wait_time=5.,
        draw_gt=True
        )
    )

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# TODO: support auto scaling lr