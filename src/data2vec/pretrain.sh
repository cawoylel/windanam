python fairseq/fairseq_cli/hydra_train.py -m \
--config-dir ./configs \
--config-name data2vec_300m.yaml \
task.data=$1 \
checkpoint.save_interval_updates=200 \
dataset.validate_interval_updates=200 \
optimization.max_epoch=6 \
distributed_training.distributed_world_size=1 \
optimization.update_freq='[16]' \
common.user_dir=examples/data2vec