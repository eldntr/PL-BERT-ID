# tools/compute_phoneme_vocab.py
import os, json, gzip
from glob import glob
from typing import Iterable, Set

STRESS_MARKS = {"ˈ", "ˌ"}  # buang kalau keep_stress=False

def _iter_json_records(path: str) -> Iterable[dict]:
    """
    Yield records dari file:
      - JSON array besar: [ {...}, {...}, ... ]
      - JSON object tunggal: { ... }
      - JSONL/NDJSON: satu JSON per baris
      - *.gz juga didukung
    """
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)

        # JSON Lines?
        if first not in ("[", "{"):
            # asumsikan JSONL/NDJSON
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
            return

        # JSON array / object
        try:
            obj = json.load(f)
        except Exception:
            return

        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    yield rec
        elif isinstance(obj, dict):
            yield obj

def _extract_phonemes_from_record(rec: dict) -> Iterable[str]:
    """
    Ambil string fonem dari dua skema umum:
      1) rec["phonemes"] -> list[str] (tiap item = string fonem per kata)
      2) rec["phonemize"] -> list[{"phoneme": str, ...}]
    Juga dukung fallback: rec["phoneme"] langsung.
    """
    if "phonemes" in rec and isinstance(rec["phonemes"], list):
        for s in rec["phonemes"]:
            if isinstance(s, str):
                yield s
    if "phonemize" in rec and isinstance(rec["phonemize"], list):
        for x in rec["phonemize"]:
            if isinstance(x, dict):
                s = x.get("phoneme")
                if isinstance(s, str):
                    yield s
    if "phoneme" in rec and isinstance(rec["phoneme"], str):
        yield rec["phoneme"]

def compute_phoneme_vocab_from_folder(root_dir: str,
                                      keep_stress: bool = True,
                                      save_dir: str = ".") -> Set[str]:
    """
    Scan rekursif semua *.json / *.jsonl / *.ndjson (dan .gz) di root_dir.
    Kumpulkan SET semua karakter fonem unik (per-char) dari seluruh shard.
    """
    exts = ("*.json", "*.jsonl", "*.ndjson", "*.json.gz", "*.jsonl.gz", "*.ndjson.gz")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(root_dir, "**", ext), recursive=True))

    phoneme_set: Set[str] = set()
    total_files = 0
    total_records = 0

    for fp in files:
        total_files += 1
        for rec in _iter_json_records(fp):
            total_records += 1
            for s in _extract_phonemes_from_record(rec):
                for ch in s:
                    if not ch.strip():
                        continue
                    if (not keep_stress) and (ch in STRESS_MARKS):
                        continue
                    phoneme_set.add(ch)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "phoneme_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(list(phoneme_set)), f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "phoneme2id.json"), "w", encoding="utf-8") as f:
        mapping = {p: i for i, p in enumerate(sorted(list(phoneme_set)))}
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Scanned files: {total_files}, records: {total_records}")
    print(f"Unique phonemes: {len(phoneme_set)} (saved to phoneme_vocab.json & phoneme2id.json)")
    return phoneme_set

if __name__ == "__main__":
    # contoh pemakaian:
    # python tools/compute_phoneme_vocab.py
    compute_phoneme_vocab_from_folder("wiki_phoneme", keep_stress=True, save_dir=".")
