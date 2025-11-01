# coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
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
        Dataset untuk input speech/text berbasis BPE tokenizer.
        Phoneme tetap di word-level, teks menggunakan subword (BPE).
        """
        self.data = dataset
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()

        self.token_separator = token_separator
        self.token_mask = token_mask

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.vocab = self.tokenizer.get_vocab()
        self.word_separator = self.tokenizer.eos_token_id  # optional
        # Pilih pad id aman untuk batching
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (self.word_separator if self.word_separator is not None else 0)

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

        phoneme_list = ''.join(phonemes)
        masked_idx_list = []

        for phoneme_word, bpe_ids in zip(phonemes, input_ids):
            # subword_tokens = self.tokenizer.convert_ids_to_tokens(bpe_ids)  # optional debug
            words.extend(bpe_ids)
            labels += phoneme_word + " "

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
                    else:
                        phoneme += phoneme_word
                else:
                    phoneme_masked = self.token_mask * len(phoneme_word)
                    start = len(phoneme)
                    phoneme += phoneme_masked
                    masked_idx_list.extend(list(range(start, start + len(phoneme_word))))
            else:
                phoneme += phoneme_word

            # pemisah kata untuk fonem + word separator untuk BPE group
            phoneme += self.token_separator
            if self.word_separator is not None:
                words.append(self.word_separator)

        mel_length = len(phoneme)

        # Truncation bila > max_mel_length (sinkronkan labels & masked indices)
        masked_index = []
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            phoneme = phoneme[random_start:random_start + self.max_mel_length]
            labels = labels[random_start:random_start + self.max_mel_length]
            for m in masked_idx_list:
                if random_start <= m < random_start + self.max_mel_length:
                    masked_index.append(m - random_start)
        else:
            masked_index = masked_idx_list

        # Cleaning ke integer list
        phoneme_clean = self.text_cleaner(phoneme)
        labels_clean = self.text_cleaner(labels)

        phonemes_tensor = torch.LongTensor(phoneme_clean)
        labels_tensor = torch.LongTensor(labels_clean)
        words_tensor = torch.LongTensor(words)

        return phonemes_tensor, words_tensor, labels_tensor, masked_index

class Collater(object):
    def __init__(self, tokenizer=None, return_wave=False, debug=False):
        self.return_wave = return_wave
        self.debug = debug
        self.text_pad_index = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
        self.word_separator = getattr(tokenizer, "eos_token_id", None) if tokenizer else None

    def __call__(self, batch):
        batch_size = len(batch)

        # Urutkan berdasarkan panjang phoneme (b[0])
        lengths = [b[0].shape[0] for b in batch]
        order = np.argsort(lengths)[::-1]
        batch = [batch[i] for i in order]

        max_seq_length = max(lengths)

        # Inisialisasi tensor padded (semua disamakan ke panjang phoneme terpanjang)
        words = torch.full((batch_size, max_seq_length), self.text_pad_index, dtype=torch.long)
        labels = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        phonemes = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

        input_lengths = []
        token_lengths = []
        masked_indices = []

        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            seq_len = min(phoneme.size(0), label.size(0), max_seq_length)

            # Batas target words ke seq_len agar CTC target_length ≤ input_length
            word_len = min(word.size(0), max_seq_length)

            phonemes[bid, :seq_len] = phoneme[:seq_len]
            labels[bid, :seq_len] = label[:seq_len]
            words[bid, :word_len] = word[:word_len]

            input_lengths.append(seq_len)
            masked_indices.append([m for m in masked_index if m < seq_len])

            word_slice = words[bid, :word_len]
            valid_mask = (word_slice != self.text_pad_index)
            if self.word_separator is not None:
                valid_mask &= (word_slice != self.word_separator)
            token_lengths.append(int(valid_mask.sum().item()))

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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu')
    )
    return loader
