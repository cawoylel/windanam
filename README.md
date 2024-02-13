# ASR models

This repo contains code to train OpenAI whisper and Meta MMS models on fula language.

# Continue pretraining

## Setup
### Install fairseq

First, install `fairseq` from source:
```shell
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
```

### Prepare the data

```shell
python fairseq/examples/wav2vec/wav2vec_manifest.py $AUDIO_FOLDER --dest prepared --ext wav --valid-percent 0.1
```

## data2vec

You can just run the continue pretraining as:

```shell
python fairseq/fairseq_cli/hydra_train.py -m \
--config-dir configs \
--config-name data2vec_300m.yaml \
task.data=prepared\
distributed_training.distributed_world_size=1 \
optimization.update_freq='[16]' \
common.user_dir=examples/data2vec
```

## MMS

### Download the model

For downloading the large model:
```shell
wget https://dl.fbaipublicfiles.com/mms/pretraining/base_1b.pt
```

For downloading the base model:
```shell
wget https://dl.fbaipublicfiles.com/mms/pretraining/base_300m.pt
```

Where ``$AUDIO_FOLDER`` is the folder containing the wav audio files. The audios have to be sampled to 16k and must contain only a single channel.

### Run the continue pretraining

For the base mms 300m:
```shell
fairseq-hydra-train \
    checkpoint.restore_file=base_300m.pt \
    checkpoint.reset_dataloader=True \
    checkpoint.reset_lr_scheduler=True \
    checkpoint.reset_optimizer=True \
    checkpoint.save_interval_updates=200 \
    task.data=prepared \
    dataset.validate_interval_updates=200 \
    optimization.max_epoch=6 \
    optimization.lr=[0.000056] \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq=["16"] \
    lr_scheduler.warmup_updates=200 \
    --config-dir configs/ \
    --config-name mms_300m
```



