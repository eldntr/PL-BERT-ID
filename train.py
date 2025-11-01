# coding: utf-8
import os
import os.path as osp
import math
import time
import yaml
import shutil
import random
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from datasets import load_from_disk
from transformers import AutoTokenizer, AlbertConfig, AlbertModel, get_cosine_schedule_with_warmup

# ====== Impor model kamu ======
# Pastikan salah satu dari dua baris ini berhasil.
try:
    from model import MultiTaskModel  # sesuaikan dengan proyekmu
except Exception:
    from models import MultiTaskModel  # fallback kalau paketnya berbeda

# ====== Impor dataloader (punyamu) ======
from dataloader import build_dataloader

# =========================================
# Utils
# =========================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def length_to_mask(lengths: torch.Tensor, max_len: int = None) -> torch.BoolTensor:
    """
    lengths: (B,) panjang valid per sampel
    return : (B, T_max) boolean mask True untuk posisi valid
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    rng = torch.arange(max_len, device=lengths.device)[None, :]  # (1, T)
    return rng < lengths[:, None]  # (B, T)

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================
# Training
# =========================================
def train(config_path: str = "config.yml"):
    # -----------------------------
    # Load config
    # -----------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = bool(config.get("mixed_precision", True))

    # -----------------------------
    # Paths & logging
    # -----------------------------
    log_dir = config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(osp.join(log_dir, "train.log"), mode="a")
        ]
    )
    logger = logging.getLogger("train")

    # -----------------------------
    # Dataset & DataLoader
    # config["data_folder"]   -> path dataset (HF datasets.load_from_disk)
    # config["batch_size"]
    # config["dataset_params"] -> dict utk FilePathDataset
    # -----------------------------
    logger.info("Loading dataset from disk ...")
    dataset = load_from_disk(config["data_folder"])

    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 0))

    train_loader = build_dataloader(
        dataset,
        validation=False,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        dataset_config=config.get("dataset_params", {}),
        collate_config={}  # Collater kamu sudah dipakai di dalam build_dataloader
    )

    # Ambil tokenizer dari dataset (melalui FilePathDataset)
    # build_dataloader membuat FilePathDataset internal dan Collater dengan tokenizer tsb.
    # Untuk sinkronisasi, kita inisialisasi tokenizer lagi dengan nama yg sama:
    tk_name = config.get("dataset_params", {}).get(
        "tokenizer", "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
    )
    tokenizer = AutoTokenizer.from_pretrained(tk_name)

    # -----------------------------
    # Model
    # config["model_params"]: dict utk AlbertConfig, dsb.
    # model mengeluarkan dua head:
    #   tokens_pred: (B, T, V_token)  -> CE loss
    #   words_pred : (B, T, V_ctc)    -> CTC loss (sebelum log_softmax)
    # -----------------------------
    logger.info("Building model ...")
    albert_conf = AlbertConfig(**config["model_params"])
    backbone = AlbertModel(albert_conf)
    model = MultiTaskModel(backbone, **config.get("head_params", {})).to(device)

    logger.info(f"Trainable params: {count_parameters(model):,}")

    # -----------------------------
    # Optimizer & Scheduler
    # -----------------------------
    learning_rate = float(config.get("learning_rate", 5e-5))
    weight_decay = float(config.get("weight_decay", 0.0))
    num_steps = int(config.get("num_steps", 10000))
    warmup_steps = int(config.get("warmup_steps", 1000))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps
    )

    # -----------------------------
    # Losses
    # -----------------------------
    # CTC: blank=0 dan zero_infinity untuk stabilitas
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    # CE: reduction=mean
    ce_loss_fn = nn.CrossEntropyLoss()

    # Loss weights
    loss_weights = config.get("loss_weights", {
        "ctc_before": 1.0,
        "token_before": 0.0,
        "ctc_after": 1.0,
        "token_after": 1.0
    })
    ctc_weight = float(loss_weights.get("ctc_before", 1.0))
    token_weight = float(loss_weights.get("token_before", 0.0))

    # Sanity check dimensi head token
    expected_token_vocab = int(config["model_params"]["vocab_size"])
    # (Pastikan head CE model kamu pakai dim terakhir = expected_token_vocab)

    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and mixed_precision))

    # -----------------------------
    # Training loop
    # -----------------------------
    global_step = 0
    model.train()

    logger.info("Start training ...")
    start_time = time.time()

    while global_step < num_steps:
        for batch in train_loader:
            if global_step >= num_steps:
                break

            # Collater (versi perbaikan) mengembalikan:
            # words, labels, phonemes, input_lengths, masked_indices, token_lengths, target_lengths
            (
                words,
                labels,
                phonemes,
                input_lengths,
                masked_indices,
                token_lengths,
                target_lengths_ce
            ) = batch

            words = words.to(device)         # (B, T)
            labels = labels.to(device)       # (B, T), id untuk CE head
            phonemes = phonemes.to(device)   # (B, T), id/time-steps untuk encoder

            # panjang input (time steps) per sampel
            phoneme_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)  # (B,)

            # attention_mask: 1=valid, 0=pad (BENAR, jangan dibalik)
            mask_bool = length_to_mask(phoneme_lengths, max_len=phonemes.size(1))  # (B, T) bool
            attention_mask = mask_bool.to(dtype=torch.long)  # (B, T)

            with torch.cuda.amp.autocast(enabled=(device_type == "cuda" and mixed_precision)):
                # Forward; model harus mengembalikan dua head:
                #   tokens_pred: (B, T, V_token)  -> CE
                #   words_pred : (B, T, V_ctc)    -> CTC
                tokens_pred, words_pred = model(phonemes, attention_mask=attention_mask)

                # ---------- CTC ----------
                # words_pred -> (B, T, V_ctc) → log_probs (T, B, V_ctc)
                log_probs = F.log_softmax(words_pred, dim=-1).transpose(0, 1)  # (T, B, C)

                # Siapkan target CTC dari 'words' (BPE) tanpa pad/eos
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                eos_id = tokenizer.eos_token_id

                valid_mask_ctc = (words != pad_id)
                if eos_id is not None:
                    valid_mask_ctc &= (words != eos_id)

                # Per-sampel panjang target CTC
                target_lengths_tensor = torch.tensor(
                    token_lengths, device=device, dtype=torch.long
                )  # (B,)

                # !!! Penting: input_lengths untuk CTC adalah panjang time-steps (dari encoder input)
                input_lengths_tensor = phoneme_lengths  # (B,)

                # Filter sampel tidak valid: target_len==0 atau target_len>input_len
                keep = (target_lengths_tensor > 0) & (input_lengths_tensor >= target_lengths_tensor)
                if torch.any(~keep):
                    # Filter log_probs (dim=1 = batch)
                    log_probs = log_probs[:, keep, :]
                    input_lengths_tensor = input_lengths_tensor[keep]
                    target_lengths_tensor = target_lengths_tensor[keep]

                # Rebangun targets sesuai 'keep'
                if keep.ndim == 0:
                    # edge case (jarang)
                    keep_words = words.unsqueeze(0)
                    keep_mask = valid_mask_ctc.unsqueeze(0)
                else:
                    keep_words = words[keep]
                    keep_mask = valid_mask_ctc[keep]

                # Concatenate semua target valid per-batch
                if keep_words.size(0) == 0:
                    # Tidak ada sampel valid di batch ini
                    global_step += 1
                    continue

                targets_list = [wk[vk] for wk, vk in zip(keep_words, keep_mask)]
                if len(targets_list) == 0:
                    global_step += 1
                    continue
                targets = torch.cat(targets_list, dim=0)

                # Guard batch kosong setelah filtering
                T_cur, N_cur = log_probs.size(0), log_probs.size(1)
                if targets.numel() == 0 or N_cur == 0:
                    global_step += 1
                    continue

                loss_vocab = ctc_loss_fn(
                    log_probs,                # (T, B, C)
                    targets,                  # (sum target_lengths)
                    input_lengths_tensor,     # (B,)
                    target_lengths_tensor     # (B,)
                )

                # ---------- CE (token head) ----------
                # tokens_pred: (B, T, V_token)
                V_tok = tokens_pred.size(-1)
                if V_tok != expected_token_vocab:
                    raise ValueError(f"Token-head dim mismatch: model={V_tok} vs config={expected_token_vocab}")

                loss_token = torch.tensor(0.0, device=device)
                count = 0
                # Iter per-sampel untuk pakai masked_indices
                for pred, lbl, L, masked in zip(tokens_pred, labels, input_lengths, masked_indices):
                    L = int(L)
                    if L <= 0 or len(masked) == 0:
                        continue
                    idx = torch.as_tensor(masked, dtype=torch.long, device=device)
                    idx = idx[idx < L]  # batasin di panjang valid
                    if idx.numel() == 0:
                        continue

                    span = lbl[:L][idx]  # (K,)
                    # Buang label di luar rentang (jaga stabilitas)
                    keep_ce = (span >= 0) & (span < V_tok)
                    if keep_ce.sum().item() == 0:
                        continue
                    span = span[keep_ce]
                    idx = idx[keep_ce]

                    loss_token += ce_loss_fn(pred[:L][idx], span)  # CE per posisi
                    count += 1
                if count > 0:
                    loss_token = loss_token / count

                # ---------- weighting ----------
                if global_step >= warmup_steps:
                    ctc_weight = float(loss_weights.get("ctc_after", 1.0))
                    token_weight = float(loss_weights.get("token_after", 1.0))

                loss = ctc_weight * loss_vocab + token_weight * loss_token

            # -----------------------------
            # Backprop
            # -----------------------------
            optimizer.zero_grad(set_to_none=True)

            if device_type == "cuda" and mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % int(config.get("log_interval", 50)) == 0:
                logger.info(
                    f"step={global_step} | loss={loss.item():.4f} "
                    f"(ctc={loss_vocab.item():.4f}, ce={loss_token.item():.4f}) | "
                    f"ctc_w={ctc_weight:.2f}, ce_w={token_weight:.2f}"
                )

            if global_step % int(config.get("ckpt_interval", 1000)) == 0:
                ckpt_path = osp.join(log_dir, f"model_step_{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": config,
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    logger.info(f"Training finished: steps={global_step}, time={total_time/60:.2f} min")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    args = parser.parse_args()
    train(args.config)
