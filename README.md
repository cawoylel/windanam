# ASR models

This repo contains code to train OpenAI whisper and Meta MMS models on fula language.

## MMS
### Continue pretraining

#### Download the model

```shell
wget https://dl.fbaipublicfiles.com/mms/pretraining/base_1b.pt
```

#### Install fairseq

First, install `fairseq` from source:
```shell
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
```

#### Prepare the data

```shell
python fairseq/examples/wav2vec/wav2vec_manifest.py $AUDIO_FOLDER --dest prepared --ext wav --valid-percent 0.1
```

Where ``$AUDIO_FOLDER`` is the folder containing the audio files. The audios have to be sampled to 16k and must contain only a single channel.

#### Run the continue pretraining

```shell
sh src/mms/continue_pretraining.sh
```



