python fairseq/fairseq_cli/hydra_train.py -m \
--config-dir windanam/configs \
--config-name data2vec_300m.yaml \
dataset.batch_size=1 \
task.data=prepared \
dataset.validate_interval_updates=200 \
optimization.max_update=40_000 \
distributed_training.distributed_world_size=1 \
optimization.update_freq='[16]' \
common.user_dir=examples/data2vec