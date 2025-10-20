# PL-BERT v2 — Training Overview (Multilingual Subword BPE)

This document summarizes the core design differences between the Author implementation (word-level, English) and PL-BERT v2 (multilingual subword BPE). It covers training objectives, data pipeline design, model initialization, training loop updates, and the combined loss strategy (CTC + CrossEntropy).

## 1. Goals and Scope

- PL-BERT (Author, word-level English)
  - Goal: learn a word-to-phoneme mapping in English with CrossEntropyLoss.
  - Tokenization: word level.
  - Target: 1 word ↔ 1 token.
  - Components: AlbertModel + MultiTaskModel.
  - Optimizer: AdamW.
  - Mixed precision: 🤗 Accelerate.
  - Loss: CrossEntropyLoss only.
  - Checkpointing: manual `torch.save`.

- PL-BERT v2 (Multilingual subword BPE)
  - Goal: learn multilingual phoneme-to-subword mappings with CTCLoss + CrossEntropyLoss.
  - Tokenization: multilingual subword BPE (English/Indonesian).
  - Target: 1 phoneme sequence ↔ N subword tokens.
  - Components: AlbertModel + CTC-aware MultiTaskModel.
  - Optimizer: AdamW (identical setup).
  - Mixed precision: 🤗 Accelerate (explicit `split_batches=True`).
  - Loss: CTCLoss (sequence) + CrossEntropyLoss (masked token).
  - Checkpointing: `accelerator.save()` (safe for DDP/multi-GPU).

## 2. Training Pipeline Architecture

```mermaid
flowchart TD
  subgraph Author
    A1[Dataset (word-level)] --> B1[Albert + MultiTaskModel]
    B1 --> C1[words_pred / tokens_pred]
    C1 --> D1[CrossEntropy Loss (vocab & token)]
  end

  subgraph PL-BERT v2
    A2[Dataset (BPE subword + phoneme)] --> B2[Albert + MultiTaskModel]
    B2 --> C2[words_pred (logits for CTC)]
    B2 --> C3[tokens_pred (masked LM)]
    C2 --> D2[CTCLoss (vocab)]
    C3 --> D3[CrossEntropy Loss (token)]
    D2 & D3 --> E2[Total Loss = CTC + Token]
  end
```

## 3. Data Processing Stages

- Dataset loading: `load_from_disk(config["data_folder"])` (shared).
- Dataloader:
  - Author: `build_dataloader(dataset, dataset_config)`.
  - PL-BERT v2: updated collater adds `token_lengths` and dynamic padding.
- Tokenizer:
  - Author: Transfo-XL (`transfo-xl-wt103`).
  - PL-BERT v2: Hugging Face `AutoTokenizer` (LLaMA-3 BPE).
- Vocabulary size:
  - Author: derived from `token_maps.pkl`.
  - PL-BERT v2: `tokenizer.vocab_size + 1` (CTC blank token at index 0).
- Separator token:
  - Author: hard-coded id `3039`.
  - PL-BERT v2: `eos_token_id` from the tokenizer.
- Padding:
  - Author: fixed `0`.
  - PL-BERT v2: dynamic `tokenizer.pad_token_id`.

## 4. Model Initialization

- Author
  - `num_vocab` computed from the token map file (manual).

- PL-BERT v2
  - `ctc_output_dim = tokenizer.vocab_size + 1` (blank index 0 reserved for CTC).
  - Example:
    ```
    ctc_output_dim = tokenizer.vocab_size + 1
    bert = MultiTaskModel(
      bert,
      num_vocab = ctc_output_dim,
      num_tokens = config['model_params']['vocab_size'],
      hidden_size = config['model_params']['hidden_size']
    )
    ```

## 5. Training Loop Differences

- Batch inputs:
  - Author: `(words, labels, phonemes, input_lengths, masked_indices)`.
  - PL-BERT v2: `(words, labels, phonemes, input_lengths, masked_indices, token_lengths)`.
- Attention mask:
  - Author: `text_mask = length_to_mask(input_lengths)`.
  - PL-BERT v2: same, additionally pushed to device (phoneme-length driven mask).
- Forward pass:
  - Both: `tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())`.
- Vocabulary loss:
  - Author: mean CrossEntropy over vocabulary logits.
  - PL-BERT v2: `CTCLoss(log_probs, targets, input_lengths, target_lengths)` for stability across length mismatches.
- Token targets:
  - Author: `words[:_text_length]`.
  - PL-BERT v2: `torch.masked_select(words, valid_mask).long() + 1` (offset for blank id 0).
- Token loss:
  - Both: CrossEntropy on masked positions; PL-BERT v2 skips computation if `masked_indices` is empty.
- Total loss:
  - Both: `loss = loss_vocab + loss_token`.
- Checkpointing:
  - Author: `torch.save(state, path)`.
  - PL-BERT v2: `accelerator.save()` (DDP-safe).

### Loop snippet (condensed)
- Author (word-level)
```
for _, batch in enumerate(train_loader):
  words, labels, phonemes, input_lengths, masked_indices = batch
  text_mask = length_to_mask(torch.Tensor(input_lengths))
  tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())

  loss_vocab = mean(CrossEntropy(words_pred, words))
  loss_token = mean(CrossEntropy(tokens_pred, labels[masked_indices]))
  loss = loss_vocab + loss_token
```

- PL-BERT v2 (subword + CTC)
```
for _, batch in enumerate(train_loader):
  words, labels, phonemes, input_lengths, masked_indices, token_lengths = batch
  tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())

  log_probs = F.log_softmax(words_pred, dim=-1).transpose(0, 1)
  targets = torch.masked_select(words, valid_mask).to(torch.long) + 1

  loss_vocab = ctc_criterion(log_probs, targets, input_lengths_tensor, target_lengths_tensor)
  loss_token = mean(CrossEntropy(tokens_pred[masked_indices], labels[masked_indices]))
  loss = loss_vocab + loss_token
```

Note: CTC enables alignment-free learning between phoneme sequences and subword tokens without explicit alignment metadata.

## 6. Loss Architecture (CTC + Token CrossEntropy)

- Components and purpose:
  - CTC head: aligns phoneme → subword sequences (`CTCLoss`).
  - Token head: masked-language-model predictions (`CrossEntropyLoss`).

- Total loss:
  - `L_total = L_ctc + L_token`.

```mermaid
flowchart TD
  A[Input Phoneme Sequence (batch, L)] --> B[Albert Encoder]
  B --> C[Hidden States H (batch×L×hidden)]
  C --> D1[CTC Head (Linear)]
  C --> D2[Token Head (Linear)]
  D1 --> E1[log_probs = log_softmax(words_pred)]
  D2 --> E2[tokens_pred (mask prediction)]
  E1 --> F1[CTCLoss(log_probs, targets, input_lengths, target_lengths)]
  E2 --> F2[CrossEntropyLoss(tokens_pred, labels[masked_indices])]
  F1 & F2 --> G[Total Loss = L_ctc + L_token]
```

- CTC implementation details:
  - Logits for CTC shaped `[seq_len, batch, vocab_size]` (after transpose).
  - Blank token: index `0`.
  - `zero_infinity=True` recommended for numerical stability.

- Key tensors (example shapes):
  - `phonemes`: `[B, L]`
  - `words_pred`: `[B, L, V]`
  - `tokens_pred`: `[B, L, V_token]`
  - `log_probs`: `[L, B, V]`
  - `targets`: `[sum(target lengths per batch)]`
  - `input_lengths`: `[B]`
  - `target_lengths`: `[B]`

## 7. Technical and Design Implications

- CTC loss handles varying phoneme–subword sequence lengths without manual alignment.
- Bilingual data: trains on mixed English–Indonesian corpora via BPE tokenizer + automatic phonemizer.
- Logging: track `vocab_loss` (CTC) and `token_loss` (CE) separately for diagnostics.
- Padding/separators: masking `pad_id` and `eos_token` prevents spurious loss contributions.
- Stability: dual-loss setup balances phonetic alignment and token-level predictions.
- Runtime: heavier than Author, but more robust for multilingual scenarios.

## 8. Transformation Summary (Author → PL-BERT v2)

- Token alignment:
  - Author: 1:1 (word ↔ phoneme).
  - PL-BERT v2: N:1 (subword ↔ phoneme).
- Language coverage:
  - Author: English only.
  - PL-BERT v2: Multilingual (English/Indonesian).
- Loss formulation:
  - Author: CrossEntropy.
  - PL-BERT v2: CTCLoss + CrossEntropy.
- Tokenizer:
  - Author: Transfo-XL.
  - PL-BERT v2: LLaMA-3 BPE (`AutoTokenizer`).
- Dataloader:
  - Author: word-level batching.
  - PL-BERT v2: BPE-aware batching + `token_lengths`.
- Checkpoint I/O:
  - Author: manual serialization.
  - PL-BERT v2: 🤗 Accelerate (DDP-safe).
- Debug/logging:
  - Author: minimal.
  - PL-BERT v2: detailed (separate losses, masking stats).
- Robustness:
  - Author: OOV-prone.
  - PL-BERT v2: resilient to OOV and code-switching.
- Alignment handling:
  - Author: explicit alignment.
  - PL-BERT v2: implicit via CTC.

## 9. Conclusion

PL-BERT v2 extends the Author implementation for multilingual, variable-length training scenarios:
- Replaces word-level supervision with subword alignment powered by CTC.
- Enables training on mixed-language text with flexible sequence lengths.
- Integrates Hugging Face tokenizers and the 🤗 Accelerate pipeline end-to-end.

The Author implementation remains efficient for monolingual word-level experiments but lacks the flexible alignment and multilingual generalization unlocked by PL-BERT v2.
