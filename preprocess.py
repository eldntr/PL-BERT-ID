# preprocess_ctc_bpe.py
import os
import yaml
import pickle
from tqdm import tqdm

from datasets import load_dataset, load_from_disk, concatenate_datasets
from phonemize import phonemize
import phonemizer
from phonemize_data import EnIndPhonemizer
from transformers import AutoTokenizer

config_path = "Configs/config.yml"
config = yaml.safe_load(open(config_path))

global_phonemizer = EnIndPhonemizer(ipa=True, keep_stress=False)

tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'])

root_directory = "./wiki_phoneme"
num_shards = 100
max_workers = 32

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
    
def process_shard(i):
    shard_dir = f"{root_directory}/shard_{i}"
    if os.path.exists(shard_dir):
        print(f"Shard {i} already exists!")
        return
    shard = dataset.shard(num_shards=num_shards, index=i)
    print(f"Processing shard {i}...")
    processed = shard.map(lambda t: phonemize(t['text'], global_phonemizer, tokenizer),
                          remove_columns=['text'])
    os.makedirs(shard_dir, exist_ok=True)
    processed.save_to_disk(shard_dir)

from pebble import ProcessPool
from concurrent.futures import TimeoutError

with ProcessPool(max_workers=max_workers) as pool:
    pool.map(process_shard, range(num_shards), timeout=300)

outputs = [d for d in os.listdir(root_directory)
            if os.path.isdir(os.path.join(root_directory, d))]

datasets_list = []
for o in tqdm(outputs):
    try:
        ds = load_from_disk(os.path.join(root_directory, o))
        datasets_list.append(ds)
        print(f"{o} loaded.")
    except Exception:
        continue

dataset = concatenate_datasets(datasets_list)
dataset.save_to_disk(config["data_folder"])
print(f"Dataset saved to {config['data_folder']}")

unique_phonemes = set()
for ex in tqdm(dataset):
    unique_phonemes.update(ex["phonemes"])  
    
unique_phonemes = sorted(list(unique_phonemes))
print(f"Total unique phoneme symbols: {len(unique_phonemes)}")

special_tokens = ["<pad>", "<unk>", "[M]"]
phoneme_vocab = special_tokens + unique_phonemes

token_maps = {
    p: {"word": p, "token": i}
    for i, p in enumerate(phoneme_vocab)
}

token_maps_path = config['dataset_params']['token_maps']
token_maps_dir = os.path.dirname(token_maps_path)

# Hanya buat directory jika path mengandung directory
if token_maps_dir:
    os.makedirs(token_maps_dir, exist_ok=True)

with open(token_maps_path, 'wb') as f:
    pickle.dump(token_maps, f)

print(f"Token map saved to {token_maps_path}")
print(f"Vocab size: {len(token_maps)}")