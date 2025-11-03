#coding: utf-8
import os
import os.path as osp
import random
import numpy as np
import pickle
import torch
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

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.vocab = self.tokenizer.get_vocab()
        self.word_separator = self.tokenizer.eos_token_id  # optional separator token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        phonemes = item['phonemes']      # contoh: ['sˌaja', 'ˈandʒiŋ']
        input_ids = item['input_ids']    # contoh: [[128000, 82, 12874], [53191, 287]]

        words = []
        labels = ""
        phoneme = ""
        masked_idx_list = []

        phoneme_list = ''.join(phonemes)

        for phoneme_word, bpe_ids in zip(phonemes, input_ids):
            words.extend(bpe_ids)
            labels += phoneme_word + " "

            # Masking phoneme di level kata
            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        # randomize phoneme chars
                        phoneme_rand = ''.join([
                            phoneme_list[np.random.randint(0, len(phoneme_list))]
                            for _ in range(len(phoneme_word))
                        ])
                        phoneme += phoneme_rand
                    else:
                        phoneme += phoneme_word
                else:
                    phoneme_masked = self.token_mask * len(phoneme_word)
                    phoneme += phoneme_masked
                    masked_idx_list.extend(
                        np.arange(len(phoneme) - len(phoneme_word), len(phoneme)).tolist()
                    )
            else:
                phoneme += phoneme_word
            phoneme += self.token_separator
            words.append(self.word_separator)

        mel_length = len(phoneme)
        masked_index = []
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            phoneme = phoneme[random_start:random_start + self.max_mel_length]
            labels = labels[random_start:random_start + self.max_mel_length]
            for m in masked_idx_list:
                if m >= random_start and m < random_start + self.max_mel_length:
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

        # Urutkan berdasarkan panjang phoneme
        lengths = [b[0].shape[0] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        sorted_batch = [batch[i] for i in batch_indexes]
        batch = sorted_batch

        max_seq_length = max(lengths)

        words = torch.full((batch_size, max_seq_length), self.text_pad_index, dtype=torch.long)
        labels = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        phonemes = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

        input_lengths = []
        token_lengths = []
        masked_indices = []

        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            seq_len = min(phoneme.size(0), label.size(0), max_seq_length)

            # filter EOS separator dari target
            if self.word_separator is not None:
                word = word[word != self.word_separator]

            word_len = min(word.size(0), max_seq_length)

            phonemes[bid, :seq_len] = phoneme[:seq_len]
            labels[bid, :seq_len] = label[:seq_len]
            words[bid, :word_len] = word[:word_len]

            input_lengths.append(seq_len)
            masked_indices.append(masked_index)

            valid_mask = word != self.text_pad_index
            token_lengths.append(int(valid_mask.sum().item()))

        if self.debug:
            print(f"[DEBUG] batch first input_len: {input_lengths[:3]}")
            print(f"[DEBUG] batch first token_len: {token_lengths[:3]}")
            print(f"[DEBUG] first phoneme IDs: {phonemes[0, :15].tolist()}")
            print(f"[DEBUG] first word IDs: {words[0, :15].tolist()}")

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
