fairseq-hydra-train \
    checkpoint.restore_file=base_1b.pt \
    checkpoint.reset_dataloader=True \
    checkpoint.reset_lr_scheduler=True \
    checkpoint.reset_optimizer=True \
    checkpoint.save_interval_updates=200 \
    task.data=prepared/ \
    dataset.validate_interval_updates=200 \
    optimization.max_epoch=1 \
    distributed_training.distributed_world_size=1 \
    --config-dir configs/ \
    --config-name mms_1b.yaml