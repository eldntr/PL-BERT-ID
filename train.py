# train.py
import os
import os.path as osp
import shutil
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs
from torch.optim import AdamW  

from transformers import AlbertConfig, AlbertModel, AutoTokenizer

from datasets import load_from_disk

# === proyekmu ===
from model import MultiTaskModel
from dataloader import build_dataloader  # pastikan nama file sesuai
from utils import length_to_mask

import yaml
import pickle

# ----------------------------
# Config
# ----------------------------
config_path = "Configs/config.yml"
config = yaml.safe_load(open(config_path))

data_folder     = config["data_folder"]
log_dir         = config["log_dir"]
batch_size      = config["batch_size"]
num_steps       = config["num_steps"]
log_interval    = config["log_interval"]
save_interval   = config["save_interval"]
mixed_precision = config.get("mixed_precision", "no")  # "no" | "fp16" | "bf16"

# ----------------------------
# Tokenizer (BPE)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'])

# ----------------------------
# Phoneme token maps
# ----------------------------
with open(config['dataset_params']['token_maps'], 'rb') as handle:
    token_maps = pickle.load(handle)

# phoneme vocab size (head untuk MLM/phoneme)
# ambil max id token_maps + 1 (diasumsikan token id 0..max)
phoneme_vocab_size = 1 + max([m['token'] for m in token_maps.values()])

# grapheme/BPE vocab size untuk CTC head:
# +1 untuk blank (CTC blank-id=0), sehingga semua target nanti di-shift +1
bpe_vocab_size_ctc = tokenizer.vocab_size + 1  # [0]=blank, [1..] id BPE + 1

# ----------------------------
# Loss functions
# ----------------------------
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # blank = 0
ce_loss_fn  = nn.CrossEntropyLoss()  # untuk MLM (phoneme masked tokens)

# ----------------------------
# Build / Resume
# ----------------------------
def maybe_resume(model, optimizer, accelerator, log_dir):
    """
    Cari checkpoint terakhir bertipe step_*.t7 dan load.
    Mengembalikan (iters, resumed_bool)
    """
    if not osp.exists(log_dir):
        return 0, False

    ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_") and f.endswith(".t7")]
    if len(ckpts) == 0:
        return 0, False

    # ambil terbesar
    try:
        iters = sorted([int(f.split('_')[-1].split('.')[0]) for f in ckpts])[-1]
    except Exception:
        return 0, False

    ckpt_path = osp.join(log_dir, f"step_{iters}.t7")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # state_dict pada Accelerate umumnya "module.*"
    state_dict = checkpoint['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    optimizer.load_state_dict(checkpoint['optimizer'])
    accelerator.print(f"Checkpoint loaded from {ckpt_path}")
    return iters, True

# ----------------------------
# Main train
# ----------------------------
def train():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=mixed_precision,
                              split_batches=True,
                              kwargs_handlers=[ddp_kwargs])

    # dataset HuggingFace (sudah kamu pakai di versi lama)
    raw_dataset = load_from_disk(data_folder)
    
    # Wrap dengan FilePathDataset
    from dataloader import FilePathDataset
    dataset = FilePathDataset(
        raw_dataset,
        token_maps=config['dataset_params']['token_maps'],
        word_mask_prob=0.15,
        replace_prob=0.2,
        phoneme_mask_prob=0.1,
        token_mask="[M]"
    )

    # log dir
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # simpan salinan config agar run reproducible
    try:
        shutil.copyfile(config_path, osp.join(log_dir, osp.basename(config_path)))
    except Exception:
        pass

    # dataloader CTC+BPE
    train_loader = build_dataloader(dataset,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    validation=False)

    # Model backbone (ALBERT)
    albert_conf = AlbertConfig(**config['model_params'])  # hidden_size dsb
    albert_conf.max_position_embeddings = 16384
    encoder = AlbertModel(albert_conf)

    # MultiTask head:
    # - mask_predictor: untuk MLM phoneme (dim = phoneme_vocab_size)
    # - word_predictor: untuk CTC vocab (dim = bpe_vocab_size_ctc)
    model = MultiTaskModel(
        model=encoder,
        num_tokens=phoneme_vocab_size,        # MLM head
        num_vocab=bpe_vocab_size_ctc,         # CTC head (+blank)
        hidden_size=config['model_params']['hidden_size']
    )

    optimizer = AdamW(model.parameters(), lr=config.get("lr", 1e-4))

    # Accelerate prepare
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # resume checkpoint (optional)
    iters, resumed = maybe_resume(model, optimizer, accelerator, log_dir)
    accelerator.print("Start training... (resumed)" if resumed else "Start training...")

    running_loss = 0.0
    curr_steps = 0

    model.train()
    while True:
        for batch in train_loader:
            curr_steps += 1
            iters += 1

            # ----------------------------
            # Unpack batch
            # ----------------------------
            # dataloader_ctc_bpe returns:
            # phonemes, target_phonemes, bpe_targets, input_lengths, target_lengths, masked_indices
            (phonemes,
             target_phonemes,
             bpe_targets,
             input_lengths,
             target_lengths,
             masked_indices) = batch

            # list -> tensors (on device)
            input_lengths  = torch.tensor(input_lengths, dtype=torch.long, device=phonemes.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=phonemes.device)

            # attention mask: True untuk valid, False untuk pad
            text_mask = length_to_mask(input_lengths).to(phonemes.device)  # [B, T]
            # Accelerate/BERT biasa pakai 1 untuk keep, 0 untuk mask -> invert
            attention_mask = text_mask.int()  # shape [B, T]

            # ----------------------------
            # Forward
            # ----------------------------
            tokens_pred, words_pred = model(phonemes, attention_mask=attention_mask)
            # tokens_pred: [B, T, phoneme_vocab_size]
            # words_pred : [B, T, bpe_vocab_size_ctc]

            # ----------------------------
            # MLM loss (phoneme masked)
            # ----------------------------
            # Ambil hanya indeks yang dimasker
            loss_token = 0.0
            count_masked = 0
            for i in range(tokens_pred.size(0)):  # per batch sample
                masked = masked_indices[i]
                if len(masked) == 0:
                    continue
                # ambil pred/target di posisi masked
                pred_i   = tokens_pred[i, masked]                 # [M, phoneme_vocab]
                target_i = target_phonemes[i, masked]             # [M]
                loss_token += ce_loss_fn(pred_i, target_i)
                count_masked += 1
            if count_masked > 0:
                loss_token = loss_token / count_masked
            else:
                loss_token = torch.tensor(0.0, device=phonemes.device)

            # ----------------------------
            # CTC loss (phoneme -> BPE)
            # ----------------------------
            # PyTorch CTC butuh:
            #   log_probs: [T, N, C]
            #   targets : concat 1-D (sum target lengths)
            #   input_lengths, target_lengths
            # Pastikan target BPE di-shift +1 untuk memberi ruang blank=0
            targets_concat = []
            for i in range(bpe_targets.size(0)):
                tlen = target_lengths[i].item()
                if tlen > 0:
                    # shift +1
                    targets_concat.append(bpe_targets[i, :tlen] + 1)
            if len(targets_concat) > 0:
                targets_concat = torch.cat(targets_concat, dim=0)        # [sum_tgt]
            else:
                # corner case: tidak ada target dalam batch
                targets_concat = torch.empty((0,), dtype=torch.long, device=phonemes.device)

            log_probs = words_pred.log_softmax(dim=-1).permute(1, 0, 2)  # [T, B, C]
            loss_vocab = ctc_loss_fn(log_probs, targets_concat, input_lengths, target_lengths)

            # ----------------------------
            # Total loss
            # ----------------------------
            loss = loss_token + loss_vocab

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            # (opsional) gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            # ----------------------------
            # Logging
            # ----------------------------
            if (iters % log_interval) == 0:
                avg_loss = running_loss / log_interval
                accelerator.print(
                    f"Step [{iters}/{num_steps}] "
                    f"Loss: {avg_loss:.5f} | CTC: {loss_vocab.item():.5f} | MLM: {loss_token.item():.5f}"
                )
                running_loss = 0.0

            # ----------------------------
            # Checkpoint
            # ----------------------------
            if (iters % save_interval) == 0:
                accelerator.print("Saving checkpoint...")
                state = {
                    "net": model.state_dict(),
                    "step": iters,
                    "optimizer": optimizer.state_dict(),
                }
                # gunakan accelerator.save agar 1 process yang menulis
                accelerator.save(state, osp.join(log_dir, f"step_{iters}.t7"))

            # ----------------------------
            # Stop condition
            # ----------------------------
            if curr_steps >= num_steps:
                accelerator.print("Training finished.")
                return


if __name__ == "__main__":
    train()
