#coding: utf-8

import os
import os.path as osp
import time
import random
import numpy as np
import random

import string
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from text_utils import TextCleaner

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)
 
class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,
                 tokenizer="GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
                 token_separator=" ",
                 token_mask="M",
                 max_mel_length=512,
                 word_mask_prob=0.15,
                 phoneme_mask_prob=0.1,
                 replace_prob=0.2):
        """
        Dataset untuk input speech/text berbasis BPE tokenizer (Llama v2).
        Phoneme tetap di word-level, tetapi teks menggunakan subword (BPE).
        """

        self.data = dataset
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()

        self.token_separator = token_separator
        self.token_mask = token_mask

        # ✅ Load BPE tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.vocab = self.tokenizer.get_vocab()
        self.word_separator = self.tokenizer.eos_token_id  # optional separator token

        print(f"[INFO] Vocab size: {len(self.vocab)}")
        print(f"[INFO] EOS token ID used as word separator: {self.word_separator}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        phonemes = item['phonemes']      # contoh: ['sˌaja', 'ˈandʒiŋ']
        input_ids = item['input_ids']    # contoh: [[128000, 82, 12874], [53191, 287]]

        words = []
        labels = ""
        phoneme = ""
        masked_index = []

        print(f"\n=== [IDX {idx}] INPUT DATA ===")
        print(f"Phonemes (word-level): {phonemes}")
        print(f"Input IDs (per-word BPE): {input_ids}")

        phoneme_list = ''.join(phonemes)
        masked_idx_list = []

        for i, (phoneme_word, bpe_ids) in enumerate(zip(phonemes, input_ids)):
            subword_tokens = self.tokenizer.convert_ids_to_tokens(bpe_ids)
            words.extend(bpe_ids)
            labels += phoneme_word + " "

            print(f"\n--- Word {i+1} ---")
            print(f"  phoneme_word : {phoneme_word}")
            print(f"  bpe_ids      : {bpe_ids}")
            print(f"  subword_tokens : {subword_tokens}")

            # Masking di level kata (phoneme)
            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        # randomize phoneme
                        phoneme_rand = ''.join([
                            phoneme_list[np.random.randint(0, len(phoneme_list))]
                            for _ in range(len(phoneme_word))
                        ])
                        phoneme += phoneme_rand
                        print(f"  -> phoneme randomized: {phoneme_rand}")
                    else:
                        phoneme += phoneme_word
                        print("  -> phoneme kept (replaced same word)")
                else:
                    phoneme_masked = self.token_mask * len(phoneme_word)
                    phoneme += phoneme_masked
                    print(f"  -> phoneme masked: {phoneme_masked}")
                    masked_idx_list.extend(
                        np.arange(len(phoneme) - len(phoneme_word), len(phoneme)).tolist()
                    )
            else:
                phoneme += phoneme_word
                print("  -> phoneme kept as is")

            phoneme += self.token_separator
            words.append(self.word_separator)

        mel_length = len(phoneme)
        print(f"\n[After loop] phoneme string: {phoneme}")
        print(f"[After loop] masked_idx_list: {masked_idx_list}")

        # Truncation bila mel_length > max_mel_length
        masked_index = []
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            phoneme = phoneme[random_start:random_start + self.max_mel_length]
            labels = labels[random_start:random_start + self.max_mel_length]
            for m in masked_idx_list:
                if m >= random_start and m < random_start + self.max_mel_length:
                    masked_index.append(m - random_start)
            print(f"[Truncated to max length {self.max_mel_length}]")
        else:
            masked_index = masked_idx_list

        # Cleaning ke integer list
        phoneme_clean = self.text_cleaner(phoneme)
        labels_clean = self.text_cleaner(labels)

        # ✅ Convert ke tensor
        phonemes_tensor = torch.LongTensor(phoneme_clean)
        labels_tensor = torch.LongTensor(labels_clean)
        words_tensor = torch.LongTensor(words)

        print("\n=== FINAL OUTPUT ===")
        print(f"phoneme_clean (len={len(phoneme_clean)}): {phoneme_clean}")
        print(f"labels_clean  (len={len(labels_clean)}): {labels_clean}")
        print(f"words_tensor  (len={len(words_tensor)}): {words_tensor}")
        print(f"masked_index  : {masked_index}")

        return phonemes_tensor, words_tensor, labels_tensor, masked_index   
        
class Collater(object):
    """
    Collater untuk batching dataset FilePathDataset.
    Melakukan padding ke panjang maksimum dalam batch
    dan mengembalikan semua tensor dengan urutan yang sama.
    Menyertakan log debug agar mudah melacak alignment antar sequence.
    """

    def __init__(self, tokenizer=None, return_wave=False, debug=False):
        self.return_wave = return_wave
        self.debug = debug
        # Gunakan pad_token_id dari tokenizer jika ada
        self.text_pad_index = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
        self.word_separator = getattr(tokenizer, "eos_token_id", None) if tokenizer else None

    def __call__(self, batch):
        batch_size = len(batch)

        # Urutkan berdasarkan panjang phoneme (b[0])
        lengths = [b[0].shape[0] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[i] for i in batch_indexes]

        max_seq_length = max(lengths)

        # Inisialisasi tensor padded
        words = torch.full((batch_size, max_seq_length), self.text_pad_index, dtype=torch.long)
        labels = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        phonemes = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

        input_lengths = []
        token_lengths = []
        masked_indices = []

        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            seq_len = min(phoneme.size(0), label.size(0), max_seq_length)
            word_len = min(word.size(0), max_seq_length)

            phonemes[bid, :seq_len] = phoneme[:seq_len]
            labels[bid, :seq_len] = label[:seq_len]
            words[bid, :word_len] = word[:word_len]

            input_lengths.append(seq_len)
            masked_indices.append(masked_index)

            word_slice = word[:word_len]
            valid_mask = word_slice != self.text_pad_index
            if self.word_separator is not None:
                valid_mask &= word_slice != self.word_separator
            token_lengths.append(int(valid_mask.sum().item()))

            if self.debug:
                print(f"\n[Batch {bid}] sequence len={seq_len}, word len={word_len}")
                print(f"  masked_index count={len(masked_index)}")
                print(f"  phoneme[:20] → {phoneme[:20].tolist()}")
                print(f"  label[:20]   → {label[:20].tolist()}")
                print(f"  word[:10]    → {word[:10].tolist()}")

        # Cetak ringkasan batch
        if self.debug:
            print("\n=== [COLLATE DEBUG SUMMARY] ===")
            print(f"Batch size: {batch_size}")
            print(f"Max sequence length: {max_seq_length}")
            print(f"Phonemes shape: {phonemes.shape}")
            print(f"Labels shape:   {labels.shape}")
            print(f"Words shape:    {words.shape}")
            print(f"Input lengths:  {input_lengths}")
            print(f"Masked indices: {[len(m) for m in masked_indices]}")
            print(f"Token lengths:  {token_lengths}")

        return words, labels, phonemes, input_lengths, masked_indices, token_lengths



def build_dataloader(df,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = FilePathDataset(df, **dataset_config)
    collate_fn = Collater(tokenizer=dataset.tokenizer, **collate_config)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu')
    )

    return data_loader
