import os
import os.path as osp
import shutil
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
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import wandb

from model import MultiTaskModel
from dataloader import build_dataloader
from utils import length_to_mask


# ==============================
# CONFIG & SETUP
# ==============================
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

use_interactive_backend = not matplotlib.get_backend().lower().startswith("agg")


def train():
    dataset = load_from_disk(config["data_folder"])
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(CONFIG_PATH, osp.join(log_dir, osp.basename(CONFIG_PATH)))

    log_txt_path = osp.join(log_dir, "training_log.txt")
    with open(log_txt_path, "w") as f:
        f.write("step,loss,ctc_loss,token_loss,lr\n")

    # ======== Plot setup ========
    if use_interactive_backend:
        plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    steps, losses, ctc_losses, token_losses = [], [], [], []
    line1, = ax.plot([], [], label="Total Loss", color="blue")
    line2, = ax.plot([], [], label="CTC Loss", color="red", alpha=0.6)
    line3, = ax.plot([], [], label="Token Loss", color="green", alpha=0.6)
    ax.legend(); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Training Progress"); plt.tight_layout()
    if use_interactive_backend: fig.canvas.draw()

    # ======== Dataloader ========
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

    # ======== Scheduler ========
    scheduler_cfg = config.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "").lower()
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0) or 0)
    total_steps = int(config["num_steps"])
    scheduler = None
    if scheduler_name in {"cosine_with_warmup", "cosine"}:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(warmup_steps, total_steps),
            num_training_steps=total_steps,
            num_cycles=float(scheduler_cfg.get("num_cycles", 0.5)),
        )
    elif scheduler_name in {"linear_with_warmup", "linear"}:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(warmup_steps, total_steps),
            num_training_steps=total_steps,
        )

    scaler = torch.amp.GradScaler(device=device_type, enabled=config["mixed_precision"])
    max_grad_norm = config["optimizer"]["max_grad_norm"]

    # ======== Resume checkpoint ========
    start_step = 0
    if config.get("resume", False):
        ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_")]
        if ckpts:
            try:
                last_ckpt = sorted(int(f.split("_")[-1].split(".")[0]) for f in ckpts)[-1]
                checkpoint = torch.load(osp.join(log_dir, f"step_{last_ckpt}.t7"), map_location=device)
                bert.load_state_dict(checkpoint["net"], strict=False)
                optimizer.load_state_dict(checkpoint["optimizer"])
                if scheduler and "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                start_step = checkpoint["step"]
                print(f"✅ Resumed from step {start_step}")
            except Exception as e:
                print(f"⚠️ Resume failed: {e}")

    # ======== Training setup ========
    ctc_weight = config["loss_weights"]["ctc_init"]
    token_weight = config["loss_weights"]["token_init"]
    warmup_loss_steps = config["loss_weights"]["warmup_steps"]

    num_steps = int(config["num_steps"])
    log_interval = config["log_interval"]
    save_interval = config["save_interval"]
    grad_accum_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))

    print("🚀 Start training...")
    progress = tqdm(total=num_steps, initial=start_step, dynamic_ncols=True)
    global_step = start_step
    accum_batches, accum_loss, accum_ctc, accum_token = 0, 0.0, 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch in train_loader:
        if global_step >= num_steps:
            break

        words, labels, phonemes, input_lengths, target_lengths, masked_indices, token_lengths = batch
        words, labels, phonemes = words.to(device), labels.to(device), phonemes.to(device)
        input_len_t = torch.tensor(input_lengths, dtype=torch.long, device=device)
        mask = length_to_mask(input_len_t).to(device)
        attn_mask = mask.to(torch.long)

        with torch.amp.autocast(device_type=device_type, enabled=config["mixed_precision"]):
            tokens_pred, words_pred = bert(phonemes, attention_mask=attn_mask)
            log_probs = F.log_softmax(words_pred, dim=-1).transpose(0, 1)

            # --- Flatten target untuk CTCLoss ---
            valid_mask = (words != pad_id)
            if eos_id is not None:
                valid_mask &= (words != eos_id)
            target_lengths_tensor = valid_mask.sum(dim=1).to(torch.long)
            targets = [w[m].to(device) for w, m in zip(words, valid_mask)]
            if len(targets) == 0:
                continue
            targets_concat = torch.cat(targets, dim=0)
            targets_shifted = targets_concat + 1  # +1 supaya tidak tabrakan dengan blank=0
            loss_ctc = ctc_loss_fn(log_probs, targets_shifted, input_len_t, target_lengths_tensor)

            # --- CE Loss token-level ---
            loss_token = torch.tensor(0.0, device=device)
            count = 0
            num_classes_token = bert.token_head.out_features if hasattr(bert, "token_head") else config["model_params"]["vocab_size"]
            for pred, lbl, length, masked in zip(tokens_pred, labels, input_lengths, masked_indices):
                if not masked:
                    continue
                length = int(length)
                masked_t = torch.tensor([i for i in masked if 0 <= i < length], dtype=torch.long, device=device)
                if masked_t.numel() == 0:
                    continue
                span = lbl[:length].to(device).index_select(0, masked_t)
                if span.numel() == 0 or int(span.max()) >= num_classes_token:
                    continue
                pred_masked = pred[:length].index_select(0, masked_t)
                loss_token += ce_loss_fn(pred_masked, span)
                count += 1
            if count > 0:
                loss_token /= count

            # --- Warmup dynamic loss weight ---
            if global_step >= warmup_loss_steps:
                ctc_weight = config["loss_weights"]["ctc_after"]
                token_weight = config["loss_weights"]["token_after"]

            loss = ctc_weight * loss_ctc + token_weight * loss_token

        # --- Gradient accumulate ---
        accum_batches += 1
        accum_loss += loss.item()
        accum_ctc += loss_ctc.item()
        accum_token += loss_token.item()

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
        if scheduler: scheduler.step()

        global_step += 1
        avg_loss = accum_loss / accum_batches
        avg_ctc = accum_ctc / accum_batches
        avg_token = accum_token / accum_batches
        lr = optimizer.param_groups[0]["lr"]

        with open(log_txt_path, "a") as f:
            f.write(f"{global_step},{avg_loss:.6f},{avg_ctc:.6f},{avg_token:.6f},{lr:.8f}\n")

        if wandb_cfg.get("enabled", False):
            wandb.log({
                "step": global_step,
                "loss": avg_loss,
                "ctc_loss": avg_ctc,
                "token_loss": avg_token,
                "lr": lr,
            })

        progress.set_description(f"Step {global_step}")
        progress.set_postfix(loss=f"{avg_loss:.3f}", ctc=f"{avg_ctc:.3f}", token=f"{avg_token:.3f}", lr=f"{lr:.6f}")
        progress.update(1)

        # --- Plot ---
        if global_step % config["plot"]["refresh_interval"] == 0:
            steps.append(global_step); losses.append(avg_loss)
            ctc_losses.append(avg_ctc); token_losses.append(avg_token)
            line1.set_data(steps, losses); line2.set_data(steps, ctc_losses); line3.set_data(steps, token_losses)
            ax.relim(); ax.autoscale_view()
            if use_interactive_backend:
                plt.draw(); plt.pause(0.001)
            plt.savefig(osp.join(log_dir, config["plot"]["save_path"]))
            if wandb_cfg.get("enabled", False):
                wandb.log({"plot": wandb.Image(osp.join(log_dir, config["plot"]["save_path"]))})

        # --- Save checkpoint ---
        if global_step % save_interval == 0:
            state = {"net": bert.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step}
            if scheduler: state["scheduler"] = scheduler.state_dict()
            torch.save(state, osp.join(log_dir, f"step_{global_step}.t7"))

        accum_batches, accum_loss, accum_ctc, accum_token = 0, 0.0, 0.0, 0.0

    progress.close()
    if use_interactive_backend: plt.ioff()
    plt.savefig(osp.join(log_dir, config["plot"]["save_path"]))
    print(f"✅ Training complete. Log: {log_txt_path}")


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    train()
