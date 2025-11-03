import os
import shutil
import os.path as osp
import time
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AlbertConfig,
    AlbertModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from model import MultiTaskModel
from dataloader import build_dataloader
from utils import length_to_mask
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import wandb

use_interactive_backend = not matplotlib.get_backend().lower().startswith("agg")

CONFIG_PATH = "Configs/config.yml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

device = (
    torch.device("cuda") if torch.cuda.is_available() and config["device"] in ["auto", "cuda"]
    else torch.device("cpu")
)
device_type = "cuda" if device.type == "cuda" else "cpu"

wandb_cfg = config.get("wandb", {})
if wandb_cfg.get("enabled", False):
    os.environ["WANDB_API_KEY"] = wandb_cfg["api_key"]
    wandb.init(
        project=wandb_cfg["project"],
        name=f"{config['config_name']}_{time.strftime('%Y%m%d_%H%M%S')}",
        config=config,
    )


def train():
    dataset = load_from_disk(config["data_folder"])
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(CONFIG_PATH, osp.join(log_dir, osp.basename(CONFIG_PATH)))

    log_txt_path = osp.join(log_dir, "training_log.txt")
    with open(log_txt_path, "w") as f:
        f.write("step,loss,vocab_loss,token_loss,lr\n")

    if use_interactive_backend:
        plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    steps, losses, vocab_losses, token_losses = [], [], [], []
    line1, = ax.plot([], [], label="Total Loss", color="blue")
    line2, = ax.plot([], [], label="CTC Loss", color="red", alpha=0.6)
    line3, = ax.plot([], [], label="Token Loss", color="green", alpha=0.6)
    ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    if use_interactive_backend:
        fig.canvas.draw()
    plt.tight_layout()

    train_loader = build_dataloader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        dataset_config=config["dataset_params"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config["dataset_params"]["tokenizer"])
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id
    vocab_size_ctc = tokenizer.vocab_size
    blank_id = 0
    ctc_output_dim = vocab_size_ctc + 1

    albert_cfg = AlbertConfig(**config["model_params"])
    bert = AlbertModel(albert_cfg)
    bert = MultiTaskModel(
        bert,
        num_vocab=ctc_output_dim,
        num_tokens=config["model_params"]["vocab_size"],
        hidden_size=config["model_params"]["hidden_size"],
    ).to(device)

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    ce_loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(
        bert.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    scheduler_cfg = config.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name") or config.get("lr_scheduler_type", "")
    scheduler_name = scheduler_name.lower()
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0) or 0)
    total_steps = int(config["num_steps"])
    scheduler = None

    if scheduler_name in {"cosine_with_warmup", "cosine"}:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=min(warmup_steps, total_steps),
            num_training_steps=total_steps,
            num_cycles=float(scheduler_cfg.get("num_cycles", 0.5)),
        )
    elif scheduler_name in {"linear_with_warmup", "linear"}:
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=min(warmup_steps, total_steps),
            num_training_steps=total_steps,
        )
    elif scheduler_name and scheduler_name != "none":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("T_max", max(1, total_steps))),
        )

    scaler = torch.amp.GradScaler(device=device_type, enabled=config["mixed_precision"])
    max_grad_norm = config["optimizer"]["max_grad_norm"]

    ctc_weight = config["loss_weights"]["ctc_init"]
    token_weight = config["loss_weights"]["token_init"]
    warmup_steps = config["loss_weights"]["warmup_steps"]

    num_steps = int(config["num_steps"])
    log_interval = config["log_interval"]
    save_interval = config["save_interval"]
    grad_accum_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))

    print("🚀 Start training...")
    progress = tqdm(total=num_steps, dynamic_ncols=True)
    global_step = 0

    optimizer.zero_grad(set_to_none=True)
    accum_batches, accum_loss, accum_vocab_loss, accum_token_loss = 0, 0.0, 0.0, 0.0

    for batch in train_loader:
        if global_step >= num_steps:
            break

        words, labels, phonemes, input_lengths, masked_indices, token_lengths = batch
        words, labels, phonemes = words.to(device), labels.to(device), phonemes.to(device)
        phoneme_lengths = torch.tensor(input_lengths, dtype=torch.long)
        mask = length_to_mask(phoneme_lengths).to(device)

        with torch.amp.autocast(device_type=device_type, enabled=config["mixed_precision"]):
            tokens_pred, words_pred = bert(phonemes, attention_mask=(~mask).int())
            log_probs = F.log_softmax(words_pred, dim=-1).transpose(0, 1)  # (T, N, C)
            T, N = log_probs.size(0), log_probs.size(1)

            # --- Perbaikan utama: hitung target per full sequence ---
            targets_list, target_lengths_list = [], []
            for b in range(N):
                seq = words[b]
                valid_mask = (seq != pad_id)
                if eos_id is not None:
                    valid_mask &= (seq != eos_id)
                target_seq = seq[valid_mask]
                if target_seq.numel() == 0:
                    continue
                targets_list.append(target_seq)
                target_lengths_list.append(len(target_seq))

            if len(targets_list) == 0:
                continue

            targets = torch.cat(targets_list)
            target_lengths_tensor = torch.tensor(target_lengths_list, dtype=torch.long, device=device)
            input_lengths_tensor = torch.full((len(targets_list),), T, dtype=torch.long, device=device)

            loss_vocab = ctc_loss_fn(
                log_probs[:, :len(targets_list), :], targets, input_lengths_tensor, target_lengths_tensor
            )

            # --- Token loss (mask prediction) ---
            loss_token = torch.tensor(0.0, device=device)
            count = 0
            for pred, lbl, length, masked in zip(tokens_pred, labels, input_lengths, masked_indices):
                if len(masked) > 0:
                    span = lbl[:length][masked].to(device)
                    if span.numel() == 0:
                        continue
                    loss_token += ce_loss_fn(pred[:length][masked].to(device), span)
                    count += 1
            if count > 0:
                loss_token /= count

            # --- Dynamic loss weighting ---
            if global_step >= warmup_steps:
                ctc_weight = config["loss_weights"]["ctc_after"]
                token_weight = config["loss_weights"]["token_after"]

            loss = ctc_weight * loss_vocab + token_weight * loss_token

        accum_batches += 1
        accum_loss += loss.item()
        accum_vocab_loss += loss_vocab.item()
        accum_token_loss += loss_token.item()

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        if accum_batches % grad_accum_steps != 0:
            continue

        scaler.unscale_(optimizer)
        if config["gradient_clip"]:
            torch.nn.utils.clip_grad_norm_(bert.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        avg_loss = accum_loss / accum_batches
        avg_vocab_loss = accum_vocab_loss / accum_batches
        avg_token_loss = accum_token_loss / accum_batches
        lr = optimizer.param_groups[0]["lr"]

        with open(log_txt_path, "a") as f:
            f.write(f"{global_step},{avg_loss:.6f},{avg_vocab_loss:.6f},{avg_token_loss:.6f},{lr:.8f}\n")

        if wandb_cfg.get("enabled", False):
            wandb.log({
                "step": global_step,
                "loss": avg_loss,
                "vocab_loss": avg_vocab_loss,
                "token_loss": avg_token_loss,
                "lr": lr,
            })

        progress.set_description(f"Step {global_step}")
        progress.set_postfix(
            loss=f"{avg_loss:.3f}",
            ctc=f"{avg_vocab_loss:.3f}",
            token=f"{avg_token_loss:.3f}",
            lr=f"{lr:.6f}",
        )
        progress.update(1)

        if global_step % config["plot"]["refresh_interval"] == 0:
            steps.append(global_step)
            losses.append(avg_loss)
            vocab_losses.append(avg_vocab_loss)
            token_losses.append(avg_token_loss)
            line1.set_data(steps, losses)
            line2.set_data(steps, vocab_losses)
            line3.set_data(steps, token_losses)
            ax.relim(); ax.autoscale_view()
            if use_interactive_backend:
                plt.draw(); plt.pause(0.001)
            plt.savefig(osp.join(log_dir, config["plot"]["save_path"]))
            if wandb_cfg.get("enabled", False):
                wandb.log({"plot": wandb.Image(osp.join(log_dir, config["plot"]["save_path"]))})

        if global_step % save_interval == 0:
            state = {
                "net": bert.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
            }
            if scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            torch.save(state, osp.join(log_dir, f"step_{global_step}.t7"))

        accum_batches = 0
        accum_loss = 0.0
        accum_vocab_loss = 0.0
        accum_token_loss = 0.0

    progress.close()
    if use_interactive_backend:
        plt.ioff()
    plt.savefig(osp.join(log_dir, config["plot"]["save_path"]))
    print(f"✅ Training complete. Log: {log_txt_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    train()