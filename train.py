# coding: utf-8
import os
import os.path as osp
import time
import math
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, AlbertConfig

from phoneme_tokenizer import PhonemeTokenizer
from dataloader import FilePathDataset, collater
from model import MultiTaskModel

# -----------------------------------------------------------
# (Opsional) wandb
# -----------------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ===========================================================
# Konfigurasi (silakan sesuaikan paths & hyperparams)
# ===========================================================
config = {
    # path dataset HF (load_from_disk)
    "data_folder": "wikipedia_20220301.id.processed",  # ganti

    # HF BPE tokenizer (LLaMA Indo, dsb)
    "bpe_tokenizer_name": "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",

    # tempat simpan vocab phoneme
    "phoneme_vocab_path": "phoneme_vocab.json",

    # training
    "batch_size": 1,
    "num_workers": 1,
    "device": "cpu",  # "cuda" / "cpu"
    "mlm_ratio": 0.15,
    "upsample_factor": 4,

    "max_steps": 100000,        # total update step
    "log_interval": 50,         # print setiap N step
    "save_interval": 400,      # save checkpoint setiap N step
    "save_dir": "Checkpoints",
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,

    # ALBERT config (silakan sesuaikan ukuran)
    "albert_config": {
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "max_position_embeddings": 8192,
        "type_vocab_size": 2,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "embedding_size": 256,
        "num_hidden_groups": 1,
        "inner_group_num": 1,
    },

    # wandb
    "wandb": {
        "enabled": False,
        "project": "PL-BERT-v2-ID",
        "run_name": "plbertv2_mlm_ctc_upsample"
    }
}


# ===========================================================
# Utilitas
# ===========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def build_phoneme_tokenizer(dataset, vocab_path: str) -> PhonemeTokenizer:
    """
    Jika vocab sudah ada -> load.
    Jika belum -> build dari dataset dan save.
    """
    if osp.exists(vocab_path):
        print(f"[INFO] Load phoneme vocab from {vocab_path}")
        tok = PhonemeTokenizer.load(vocab_path)
    else:
        print("[INFO] Build phoneme vocab dari dataset...")
        tok = PhonemeTokenizer.build_from_dataset(dataset)
        tok.save(vocab_path)
        print(f"[INFO] Phoneme vocab saved to {vocab_path}")
    print(f"[INFO] Phoneme vocab size: {tok.vocab_size}")
    return tok


def build_dataloader(dataset, phoneme_tokenizer, bpe_tokenizer, config):
    ds = FilePathDataset(
        dataset=dataset,
        phoneme_tokenizer=phoneme_tokenizer,
        bpe_tokenizer=bpe_tokenizer,
        mlm_ratio=config["mlm_ratio"],
    )
    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collater,
        pin_memory=(config["device"] == "cuda"),
        drop_last=True,
    )
    return loader


def build_model(ph_tok, bpe_tok, config):
    albert_cfg_dict = config["albert_config"]
    # pastikan vocab_size di config = phoneme_vocab_size
    albert_cfg_dict = dict(albert_cfg_dict)  # copy
    albert_cfg_dict["vocab_size"] = ph_tok.vocab_size

    model = MultiTaskModel.from_albert_config(
        phoneme_vocab_size=ph_tok.vocab_size,
        bpe_vocab_size=bpe_tok.vocab_size,
        albert_config_dict=albert_cfg_dict,
        pad_token_id=ph_tok.pad_token_id,
        blank_id=0,
        upsample_factor=config["upsample_factor"],
    )
    return model


def save_checkpoint(save_dir, step, model, optimizer, config):
    ensure_dir(save_dir)
    ckpt_path = osp.join(save_dir, f"step_{step}.pt")
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(payload, ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}")


# ===========================================================
# TRAIN LOOP
# ===========================================================
def train():
    set_seed(42)

    device = config["device"]
    ensure_dir(config["save_dir"])

    # -------------------------------------------------------
    # 1. Load dataset HF
    # -------------------------------------------------------
    print(f"[INFO] Loading dataset from {config['data_folder']} ...")
    # diasumsikan: setiap item punya key:
    #  - "phonemes": list of string phoneme
    #  - "input_ids": list-of-list BPE ids (dari tokenizer teks LLaMA)
    raw_ds = load_from_disk(config["data_folder"])

    # Dataloader kita ekspektasi "bpe_ids"
    # Di sini saya buat view sederhana tanpa copy besar:
    # HF Dataset support indexing -> object-nya dipass langsung ke Dataset custom.
    # Jadi kita tidak perlu buat list di memori.
    # FilePathDataset akan baca item["phonemes"] dan item["bpe_ids"],
    # maka kita tambahkan kolom baru "bpe_ids" dari "input_ids".

    # Kalau dataset punya split, misal raw_ds["train"], raw_ds["validation"]
    # Sesuaikan di sini. Untuk sederhana, kita pakai raw_ds apa adanya.
    dataset = raw_ds

    # -------------------------------------------------------
    # 2. Build tokenizers
    # -------------------------------------------------------
    ph_tok = build_phoneme_tokenizer(dataset, config["phoneme_vocab_path"])
    bpe_tok = AutoTokenizer.from_pretrained(config["bpe_tokenizer_name"])
    print(f"[INFO] BPE vocab size: {bpe_tok.vocab_size}")

    # -------------------------------------------------------
    # 3. Build DataLoader
    # -------------------------------------------------------
    train_loader = build_dataloader(dataset, ph_tok, bpe_tok, config)
    train_iter = iter(train_loader)

    # -------------------------------------------------------
    # 4. Build Model + Optimizer
    # -------------------------------------------------------
    model = build_model(ph_tok, bpe_tok, config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Loss functions
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # -------------------------------------------------------
    # 5. (Opsional) wandb init
    # -------------------------------------------------------
    if config["wandb"]["enabled"] and WANDB_AVAILABLE:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config,
        )
        wandb.watch(model)
        print("[INFO] wandb logging enabled.")
    else:
        print("[INFO] wandb disabled or not available.")

    # -------------------------------------------------------
    # 6. Training loop (step-based)
    # -------------------------------------------------------
    step = 0
    max_steps = config["max_steps"]
    log_interval = config["log_interval"]
    save_interval = config["save_interval"]

    model.train()
    t0 = time.time()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1

        X_ph = batch["X_ph"].to(device)       # (B, T_ph)
        Y_mlm = batch["Y_mlm"].to(device)     # (B, T_ph)
        Y_p2g = batch["Y_p2g"].to(device)     # (B, T_bpe)
        input_len = batch["input_lengths"].to(device)   # (B,)
        target_len = batch["target_lengths"].to(device) # (B,)

        # Forward
        logits_mlm, logits_ctc, extra = model(X_ph)

        # --- MLM Loss ---
        # NOTE: di sini MLM dihitung di semua posisi non-PAD;
        # jika ingin hanya masked positions, dataloader perlu mengirim mask_index.
        B, T_ph, V_ph = logits_mlm.shape
        mlm_logits_flat = logits_mlm.reshape(B * T_ph, V_ph)
        mlm_targets_flat = Y_mlm.reshape(B * T_ph)

        loss_mlm = mlm_loss_fn(mlm_logits_flat, mlm_targets_flat)

        # --- CTC Loss ---
        # logits_ctc: (B, T_up, V_bpe+1)
        # CTCLoss butuh shape (T_up, B, V)
        ctc_input_lengths = model.get_ctc_input_lengths(input_len)  # (B,)

        ctc_logits = logits_ctc.log_softmax(dim=-1)  # (B, T_up, V)
        ctc_logits = ctc_logits.permute(1, 0, 2)     # (T_up, B, V)

        loss_ctc = ctc_loss_fn(
            ctc_logits,
            Y_p2g,
            ctc_input_lengths,
            target_len,
        )

        loss = loss_mlm + loss_ctc

        optimizer.zero_grad()
        loss.backward()

        if config["grad_clip"] is not None and config["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()

        # ---------------------------------------------------
        # Logging
        # ---------------------------------------------------
        if step % log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()

            print(
                f"Step [{step}/{max_steps}] "
                f"Loss: {loss.item():.4f} | "
                f"MLM: {loss_mlm.item():.4f} | "
                f"CTC: {loss_ctc.item():.4f} | "
                f"dt: {dt:.2f}s"
            )

            if config["wandb"]["enabled"] and WANDB_AVAILABLE:
                wandb.log({
                    "step": step,
                    "loss": loss.item(),
                    "loss_mlm": loss_mlm.item(),
                    "loss_ctc": loss_ctc.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                })

        # ---------------------------------------------------
        # Save checkpoint
        # ---------------------------------------------------
        if step % save_interval == 0:
            save_checkpoint(config["save_dir"], step, model, optimizer, config)

    # final save
    save_checkpoint(config["save_dir"], step, model, optimizer, config)
    print("[INFO] Training finished.")


if __name__ == "__main__":
    train()
