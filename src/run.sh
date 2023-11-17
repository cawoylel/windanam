python src/train.py \
	--model_name_or_path="openai/whisper-large-v3" \
	--dataset_name="cawoylel/FulaSpeechCorpora-splited-noise_augmented" \
	--tts_dataset="cawoylel/FulaNewsTextCorporaTTS" \
	--resume_from_checkpoint="whisper-medium-tts/checkpoint-4000" \
	--language="hausa" \
	--train_split_name="train+test" \
	--eval_split_name="dev" \
	--max_steps="10000" \
	--output_dir="./whisper-medium-tts" \
	--per_device_train_batch_size="16" \
	--per_device_eval_batch_size="16" \
	--logging_steps="100" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="2000" \
	--save_strategy="steps" \
	--save_steps="2000" \
	--generation_max_length="64" \
	--preprocessing_num_workers="24" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
    --hub_model_id="cawoylel/windanam-whisper-medium_tts" \
    --hub_strategy="every_save" \
	--gradient_checkpointing \
	--group_by_length \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
    --push_to_hub