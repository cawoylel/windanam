fairseq-hydra-train \
    checkpoint.restore_file=/content/base_300m.pt \
    checkpoint.reset_dataloader=True \
    checkpoint.reset_lr_scheduler=True \
    checkpoint.reset_optimizer=True \
    checkpoint.save_interval_updates=200 \
    task.data=/content/prepared \
    dataset.validate_interval_updates=200 \
    optimization.max_epoch=6 \
    optimization.lr=[0.000056] \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq=["16"] \
    lr_scheduler.warmup_updates=200 \
    --config-dir /content/configs/ \
    --config-name mms_300m