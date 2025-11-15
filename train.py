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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

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
    "config_name": "plbertv2",  # bebas ganti nama eksperimen
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
        "api_key": "17d379f3ed0a7f3308a45e4fa92f5fba41c72dda",
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


def init_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        return True
    return False


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def get_device(cfg):
    device_str = cfg.get("device", "cpu")
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_str)


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


def build_dataloader(dataset, phoneme_tokenizer, bpe_tokenizer, config, distributed=False):
    ds = FilePathDataset(
        dataset=dataset,
        phoneme_tokenizer=phoneme_tokenizer,
        bpe_tokenizer=bpe_tokenizer,
        mlm_ratio=config["mlm_ratio"],
    )

    sampler = DistributedSampler(ds, shuffle=True) if distributed else None
    device_name = config.get("device_actual") or config.get("device", "cpu")
    pin_memory = str(device_name).startswith("cuda") and torch.cuda.is_available()

    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config["num_workers"],
        collate_fn=collater,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return loader, sampler


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
    model_to_save = model.module if hasattr(model, "module") else model
    payload = {
        "step": step,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(payload, ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}")


# ===========================================================
# TRAIN LOOP
# ===========================================================
def train():
    distributed = init_distributed()
    rank = dist.get_rank() if distributed else 0
    set_seed(42 + rank)

    device = get_device(config)
    config["device_actual"] = str(device)
    config["distributed"] = distributed
    config["world_size"] = dist.get_world_size() if distributed else 1

    ensure_dir(config["save_dir"])

    # -------------------------------------------------------
    # 1. Load dataset HF
    # -------------------------------------------------------
    if is_main_process():
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
    if distributed:
        if rank == 0:
            ph_tok = build_phoneme_tokenizer(dataset, config["phoneme_vocab_path"])
        dist.barrier()
        if rank != 0:
            ph_tok = PhonemeTokenizer.load(config["phoneme_vocab_path"])
    else:
        ph_tok = build_phoneme_tokenizer(dataset, config["phoneme_vocab_path"])

    bpe_tok = AutoTokenizer.from_pretrained(config["bpe_tokenizer_name"])
    if is_main_process():
        print(f"[INFO] BPE vocab size: {bpe_tok.vocab_size}")

    # -------------------------------------------------------
    # 3. Build DataLoader
    # -------------------------------------------------------
    train_loader, train_sampler = build_dataloader(
        dataset,
        ph_tok,
        bpe_tok,
        config,
        distributed=distributed,
    )
    train_iter = iter(train_loader)
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # -------------------------------------------------------
    # 4. Build Model + Optimizer
    # -------------------------------------------------------
    model = build_model(ph_tok, bpe_tok, config)
    model = model.to(device)

    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
        model = DDP(model, **ddp_kwargs)

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
    wandb_cfg = config.get("wandb", {})
    wandb_active = False
    if wandb_cfg.get("enabled", False) and WANDB_AVAILABLE and is_main_process():
        api_key = wandb_cfg.get("api_key")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        default_name = f"{config.get('config_name', 'run')}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_name = wandb_cfg.get("run_name", default_name)

        wandb.init(
            project=wandb_cfg["project"],
            name=run_name,
            config=config,
        )
        watch_target = model.module if hasattr(model, "module") else model
        wandb.watch(watch_target)
        print("[INFO] wandb logging enabled.")
        wandb_active = True
    elif is_main_process():
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
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
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
        if step % log_interval == 0 and is_main_process():
            dt = time.time() - t0
            t0 = time.time()

            print(
                f"Step [{step}/{max_steps}] "
                f"Loss: {loss.item():.4f} | "
                f"MLM: {loss_mlm.item():.4f} | "
                f"CTC: {loss_ctc.item():.4f} | "
                f"dt: {dt:.2f}s"
            )

            if wandb_active:
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
        if step % save_interval == 0 and is_main_process():
            save_checkpoint(config["save_dir"], step, model, optimizer, config)

    # final save
    if is_main_process():
        save_checkpoint(config["save_dir"], step, model, optimizer, config)
        print("[INFO] Training finished.")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    train()
