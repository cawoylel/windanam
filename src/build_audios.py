"""Build the audio files from huggingface datasets."""
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
import soundfile as sf

def iter_dataset(dataset: Dataset):
    for split in dataset:
        for item in tqdm(dataset[split], desc=split):
            yield item

def write_audio(dataset: Dataset,
                output_folder: str,
                shard_size: int=500):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    current_shard = 0
    idx = 0
    for item in iter_dataset(dataset):
        audio_output_folder = output_folder / item["source"]
        audio_output_folder.mkdir(exist_ok=True, parents=True)
        if idx == shard_size:
            current_shard += idx
            idx = 0
        shard = str(current_shard).zfill(6)
        audio_output_folder = output_folder / item["source"] / shard
        audio_output_folder.mkdir(exist_ok=True, parents=True)
        audio_filename = audio_output_folder / f"{item['audio']['path']}.wav"
        sf.write(file=audio_filename,
                data=item["audio"]["array"],
                samplerate=item["audio"]["sampling_rate"])
        idx += 1

def parser_args():
    parser = ArgumentParser(description="As module for creating .wav audio files from hf datasets.")
    parser.add_argument("-d", "--data",
                        help="The path to the local or remote huggingface dataset.",
                        required=True)
    parser.add_argument("-o", "--output_folder",
                        help="Where the audio files will be stored.",
                        required=True)
    
    return parser.parse_args()

def main():
    args = parser_args()
    dataset = load_dataset(args.data)
    write_audio(dataset, args.output_folder)

if __name__ == "__main__":
    main()