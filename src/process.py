from augment import augment_dataset, get_transformations
from datasets import load_dataset, DatasetDict, concatenate_datasets

import os
import random
from tqdm import tqdm
import numpy as np
import torch
import transformers
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="hausa", task="transcribe")

SEED = 1797
do_normalize_text = True
TEXT_COLUMN_NAME = "transcription"
AUDIO_COLUMN_NAME = "audio"

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHEDSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    transformers.set_seed(SEED)

    torch.backends.cudnn.deterministic = True

set_random_seed(SEED)

def remove_adamawa(dialect):
    return dialect != "adamawa"

def is_not_empty(transcription):
    return len(transcription) > 1

def normalizer(text: str):
    return text.strip().replace("*", "")

def prepare_dataset(batch):
    # load
    audio = batch[AUDIO_COLUMN_NAME]
    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[
        0
    ]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    # process targets
    input_str = normalizer(batch[TEXT_COLUMN_NAME]).strip() if do_normalize_text else batch[TEXT_COLUMN_NAME]
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(input_str).input_ids

    return batch

dataset = load_dataset("cawoylel/FulaSpeechCorpora", cache_dir="data/datasets")

concatenated = concatenate_datasets(list(dataset.values()))
train_testdev = concatenated.train_test_split(test_size=0.20)
test_dev = train_testdev["test"].train_test_split(test_size=0.2)
train_test_dev = DatasetDict({
    "train": train_testdev["train"],
    "test": test_dev["train"],
    "dev": test_dev["test"]
})

print("Raw Dataset")
print(train_test_dev)

# filter dataset
train_test_dev["dev"] = train_test_dev["dev"].filter(remove_adamawa, num_proc=4, input_columns=["dialect"])
train_test_dev["dev"] = train_test_dev["dev"].filter(is_not_empty, num_proc=4, input_columns=["transcription"])

train_test_dev["test"] = train_test_dev["test"].filter(remove_adamawa, num_proc=4, input_columns=["dialect"])
train_test_dev["test"] = train_test_dev["test"].filter(is_not_empty, num_proc=4, input_columns=["transcription"])

train_test_dev["train"] = train_test_dev["train"].filter(remove_adamawa, num_proc=4, input_columns=["dialect"])
train_test_dev["train"] = train_test_dev["train"].filter(is_not_empty, num_proc=4, input_columns=["transcription"])

print("Filtered Dataset")
print(train_test_dev)

# augment dataset
augmented_train_test_dev = train_test_dev["train"].map(augment_dataset, num_proc=6, desc="augment train dataset")
train_test_dev["train"] = concatenate_datasets([train_test_dev["train"], augmented_train_test_dev])
train_test_dev["train"] = train_test_dev["train"].shuffle(seed=10)

print("Augmented Dataset")
print(train_test_dev)
# process dataset
train_test_dev["train"] = train_test_dev["train"].map(prepare_dataset, num_proc=6, remove_columns=['audio', 'transcription'])
train_test_dev["test"] = train_test_dev["test"].map(prepare_dataset, num_proc=6, remove_columns=['audio', 'transcription'])
train_test_dev["dev"] = train_test_dev["dev"].map(prepare_dataset, num_proc=6, remove_columns=['audio', 'transcription'])

train_test_dev.save_to_disk("data/processed")

