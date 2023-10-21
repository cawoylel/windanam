from dataclasses import dataclass
from typing import Any, Dict, List, Union

import os
import random
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import transformers
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from datasets.load import load_from_disk
import evaluate

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 1797
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
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
# include forced token in the training
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# to use gradient checkpointing
model.config.use_cache = False

processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="hausa", task="transcribe")

vectorized_datasets = load_from_disk("processed")

max_input_length = 30
min_input_length = 0

max_label_length = model.config.max_length
max_label_length

def is_audio_in_length_range(length):
    return length > min_input_length and length < max_input_length

def is_labels_in_length_range(labels):
    return len(labels) < max_label_length

vectorized_datasets = vectorized_datasets.filter(
    is_audio_in_length_range, num_proc=8, input_columns=["input_length"]
)

vectorized_datasets = vectorized_datasets.filter(
    is_labels_in_length_range, num_proc=8, input_columns=["labels"]
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # convert to tensors
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad label ids to the max length in the batch
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

# evaluate with the 'normalised' WER
do_normalize_eval = False
normalizer = BasicTextNormalizer()

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        # perhaps already normalised
        label_str = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references
        pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs/windanam_whisper-base",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=800,
    num_train_epochs=2,
    learning_rate=6.25e-6,
    weight_decay=0.01,
    gradient_checkpointing=False,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=64,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["dev"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
print(train_result)

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

eval_metrics = trainer.evaluate(metric_key_prefix="eval")

print(eval_metrics)

