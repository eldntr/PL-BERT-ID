import os, yaml, pickle, glob
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, concatenate_datasets
from phonemize import phonemize
from phonemize_data import EnIndPhonemizer
from transformers import AutoTokenizer
from pebble import ProcessPool
from concurrent.futures import TimeoutError


# ---------------- config & setup ----------------
config_path = "Configs/config.yml"
config = yaml.safe_load(open(config_path))
global_phonemizer = EnIndPhonemizer(ipa=True, keep_stress=False)

tokenizer = AutoTokenizer.from_pretrained(config["dataset_params"]["tokenizer"])
tokenizer.model_max_length = int(1e9)
tokenizer.model_input_names = ["input_ids"]

root_directory = "./wiki_phoneme"
num_shards = 100
max_workers = 32


# ---------------- dataset load ------------------
parquet_folder = "wikipedia.id"
parquet_files = glob.glob(f"{parquet_folder}/*.parquet")
dataset = load_dataset("parquet", data_files=parquet_files)
if isinstance(dataset, dict):
    dataset = dataset[list(dataset.keys())[0]]
print(f"✅ Loaded {len(dataset)} examples from {len(parquet_files)} files")


# ---------------- phonemize wrapper -------------
def phonemize_and_flatten(batch):
    phonemes, input_ids, bpe_tokens, chunk_ids = [], [], [], []
    for text in batch["text"]:
        try:
            chunks = phonemize(text, global_phonemizer, tokenizer)
        except Exception:
            continue
        for r in chunks:
            phonemes.append(r["phonemes"])
            input_ids.append(r["input_ids"])
            bpe_tokens.append(r["bpe_tokens"])
            chunk_ids.append(r["chunk_id"])
    return {
        "phonemes": phonemes,
        "input_ids": input_ids,
        "bpe_tokens": bpe_tokens,
        "chunk_id": chunk_ids,
    }


# ---------------- process shard -----------------
def process_shard(i):
    shard_dir = f"{root_directory}/shard_{i}"
    if os.path.exists(shard_dir):
        print(f"Shard {i} exists, skip.")
        return
    shard = dataset.shard(num_shards=num_shards, index=i)
    if len(shard) == 0:
        print(f"⚠️ Shard {i} kosong.")
        return
    processed = shard.map(
        phonemize_and_flatten,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
        writer_batch_size=8,
        num_proc=1,
        desc=f"Phonemizing shard {i}",
    )
    os.makedirs(shard_dir, exist_ok=True)
    processed.save_to_disk(shard_dir)
    print(f"✅ Saved shard {i} ({len(processed)} examples)")


# ---------------- run multiprocessing -----------
with ProcessPool(max_workers=max_workers) as pool:
    pool.map(process_shard, range(num_shards), timeout=300)


# ---------------- merge & save ------------------
outputs = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
datasets_list = []
for o in tqdm(outputs):
    try:
        ds = load_from_disk(os.path.join(root_directory, o))
        datasets_list.append(ds)
    except Exception as e:
        print(f"Skip {o}: {e}")

dataset = concatenate_datasets(datasets_list)
dataset.save_to_disk(config["data_folder"])
print(f"✅ Final dataset saved to {config['data_folder']}")


# ---------------- build token_maps --------------
unique_phonemes = set()
for ex in tqdm(dataset):
    unique_phonemes.update(ex["phonemes"])

phoneme_vocab = ["<pad>", "<unk>", "[M]"] + sorted(list(unique_phonemes))
token_maps = {p: {"word": p, "token": i} for i, p in enumerate(phoneme_vocab)}

token_maps_path = config["dataset_params"]["token_maps"]
token_maps_dir = os.path.dirname(token_maps_path)
if token_maps_dir:
    os.makedirs(token_maps_dir, exist_ok=True)
with open(token_maps_path, "wb") as f:
    pickle.dump(token_maps, f)
print(f"✅ Token map saved ({len(token_maps)} entries)")
