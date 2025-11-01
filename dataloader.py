#coding: utf-8
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from transformers import AutoTokenizer
from text_utils import TextCleaner

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
        self.word_separator = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        phonemes = item["phonemes"]
        input_ids = item["input_ids"]

        words = []
        phoneme_seq = ""
        masked_idx_list = []
        phoneme_list = ''.join(phonemes)

        for phoneme_word, bpe_ids in zip(phonemes, input_ids):
            words.extend(bpe_ids)
            # Masking kata phoneme
            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        phoneme_rand = ''.join([
                            phoneme_list[np.random.randint(0, len(phoneme_list))]
                            for _ in range(len(phoneme_word))
                        ])
                        phoneme_seq += phoneme_rand
                    else:
                        phoneme_seq += phoneme_word
                else:
                    phoneme_seq += self.token_mask * len(phoneme_word)
                    masked_idx_list.extend(
                        np.arange(len(phoneme_seq) - len(phoneme_word), len(phoneme_seq)).tolist()
                    )
            else:
                phoneme_seq += phoneme_word
            phoneme_seq += self.token_separator
            words.append(self.word_separator)

        # --- Sinkronisasi truncation dan koreksi indeks mask ---
        start = 0
        if len(phoneme_seq) > self.max_mel_length:
            start = 0
            phoneme_seq = phoneme_seq[start:start + self.max_mel_length]
        if len(words) > self.max_mel_length:
            words = words[:self.max_mel_length]

        # Koreksi indeks mask agar tetap dalam jendela aktif
        if start > 0 or len(phoneme_seq) == self.max_mel_length:
            masked_idx_list = [m - start for m in masked_idx_list if start <= m < start + len(phoneme_seq)]
        else:
            masked_idx_list = [m for m in masked_idx_list if m < len(phoneme_seq)]

        # --- Cleaning dan konversi ke tensor ---
        phoneme_clean = self.text_cleaner(phoneme_seq)
        phonemes_tensor = torch.LongTensor(phoneme_clean)
        labels_tensor = phonemes_tensor.clone()
        words_tensor = torch.LongTensor(words)

        return phonemes_tensor, words_tensor, labels_tensor, masked_idx_list


class Collater:
    def __init__(self, tokenizer=None):
        self.text_pad_index = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
        self.word_separator = getattr(tokenizer, "eos_token_id", None) if tokenizer else None

    def __call__(self, batch):
        batch_size = len(batch)
        lengths = [b[0].shape[0] for b in batch]
        max_len = max(lengths)

        phonemes = torch.full((batch_size, max_len), 0, dtype=torch.long)
        words = torch.full((batch_size, max_len), self.text_pad_index, dtype=torch.long)
        labels = torch.full((batch_size, max_len), 0, dtype=torch.long)

        input_lengths, target_lengths, token_lengths = [], [], []
        masked_indices = []

        for bid, (ph, wd, lb, masked_index) in enumerate(batch):
            seq_len = min(ph.size(0), lb.size(0), max_len)
            word_len = min(wd.size(0), max_len)

            phonemes[bid, :seq_len] = ph[:seq_len]
            labels[bid, :seq_len] = lb[:seq_len]
            words[bid, :word_len] = wd[:word_len]

            input_lengths.append(seq_len)
            target_lengths.append(seq_len)
            token_lengths.append(word_len)

            # Filter indeks mask agar aman
            masked = [int(i) for i in masked_index if 0 <= int(i) < seq_len]
            masked_indices.append(masked)

        return words, labels, phonemes, input_lengths, target_lengths, masked_indices, token_lengths


def build_dataloader(df, batch_size=4, num_workers=1, dataset_config={}, collate_config={}):
    dataset = FilePathDataset(df, **dataset_config)
    collate_fn = Collater(tokenizer=dataset.tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
