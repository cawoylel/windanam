python fairseq/fairseq_cli/hydra_train.py -m \
--config-dir configs \
--config-name data2vec_300m.yaml \
task.data=prepared \
distributed_training.distributed_world_size=1 \
optimization.update_freq='[16]' \
common.user_dir=examples/data2vec