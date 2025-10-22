import json
import yaml

config_path = "Configs/config.yml" # you can change it to anything else
config = yaml.safe_load(open(config_path))

from phonemize import phonemize

from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'])
except Exception as exc:
    print(f"Unable to load tokenizer '{config['dataset_params']['tokenizer']}': {exc}")
    print("Using fallback tokenizer 'indobenchmark/indobert-base-p1'")
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

from datasets import load_dataset
import glob

parquet_folder = "wikipedia.id"
parquet_files = glob.glob(f"{parquet_folder}/*.parquet")

try:
    dataset = load_dataset("parquet", data_files=parquet_files)
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        dataset = dataset[split_name]
    print("Dataset loaded successfully!")
    print(f"Found {len(parquet_files)} parquet files")
    print(dataset)
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    
root_directory = "./wiki_phoneme" # set up root directory for multiprocessor processing

import os
num_shards = 50000

def process_shard(i):
    directory = root_directory + "/shard_" + str(i)
    if os.path.exists(directory):
        print("Shard %d already exists!" % i)
        return
    print('Processing shard %d ...' % i)
    shard = dataset.shard(num_shards=num_shards, index=i)
    processed_dataset = shard.map(
        lambda t: phonemize(t['text'], tokenizer),
        remove_columns=['text'],
        load_from_cache_file=False
    )
    if not os.path.exists(directory):
        os.makedirs(directory)

    json_records = []
    for example in processed_dataset:
        words = example.get("phoneme_words", []) or []
        phonemes = example.get("phonemes", []) or []
        bpe_tokens = example.get("bpe_tokens", []) or []
        decoded_tokens = example.get("decoded_tokens", []) or []
        input_ids = example.get("input_ids", []) or []

        phoneme_pairs = []
        for idx, word in enumerate(words):
            pair = {
                "word": word,
                "phoneme": phonemes[idx] if idx < len(phonemes) else "",
            }
            if idx < len(bpe_tokens):
                pair["bpe_tokens"] = bpe_tokens[idx]
            if idx < len(decoded_tokens):
                pair["decoded"] = decoded_tokens[idx]
            if idx < len(input_ids):
                pair["input_ids"] = input_ids[idx]
            phoneme_pairs.append(pair)
        json_records.append(
            {
                "before": example.get("before", ""),
                "after": example.get("after", ""),
                "phonemize": phoneme_pairs,
            }
        )

    json_path = os.path.join(directory, "phonemize.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_records, json_file, ensure_ascii=False, indent=2)

    processed_dataset.save_to_disk(directory)
    
from pebble import ProcessPool
from concurrent.futures import TimeoutError

max_workers = 32 # change this to the number of CPU cores your machine has 

with ProcessPool(max_workers=max_workers) as pool:
    pool.map(process_shard, range(num_shards), timeout=60)

# process_shard(1)

from datasets import load_from_disk, concatenate_datasets

output = [dI for dI in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory,dI))]
datasets = []
for o in output:
    directory = root_directory + "/" + o
    try:
        shard = load_from_disk(directory)
        datasets.append(shard)
        print("%s loaded" % o)
    except:
        continue
    
dataset = concatenate_datasets(datasets)
dataset.save_to_disk(config['data_folder'])
print('Dataset saved to %s' % config['data_folder'])
