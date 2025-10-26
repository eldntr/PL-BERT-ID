#coding: utf-8
import os
import os.path as osp
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
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
        Dataset BPE (subword-level) selaras dengan fonem:
        - input_ids: flattened list[int]
        - phoneme_subtokens: list[str], panjang = len(input_ids)
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
        self.word_separator = self.tokenizer.eos_token_id  # pemisah opsional

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_ids = item.get('input_ids', [])
        bpe_phonemes = item.get('phoneme_subtokens', [])

        # Jika dataset lama: fallback kasar (tidak ideal, tapi aman jalan)
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            # flatten jika masih nested
            input_ids = [t for sub in input_ids for t in sub]

        if not bpe_phonemes or len(bpe_phonemes) != len(input_ids):
            # fallback -> salin satu fonem per subtoken (asumsi fonem kata pertama)
            words_ph = item.get('phonemes', []) or item.get('phoneme_words', [])
            ph = words_ph[0] if words_ph else "a"
            bpe_phonemes = [ph] * len(input_ids)

        # Siapkan string fonem sebagai "teks" untuk encoder
        words = list(input_ids)  # akan ditambah separator setiap subtoken
        labels_str = ""
        phoneme_str = ""
        masked_idx_list = []

        # rangkai per subtoken
        for ph in bpe_phonemes:
            apply_mask = (np.random.rand() < self.word_mask_prob)
            if apply_mask:
                if np.random.rand() < self.replace_prob:
                    # kadang replace acak sebagian (noise kecil), seringnya keep
                    if np.random.rand() < (self.phoneme_mask_prob / max(self.replace_prob, 1e-8)):
                        # randomize per-char panjang yang sama
                        if len(ph) > 0:
                            rand = ''.join(random.choice(bpe_phonemes)[0] for _ in range(len(ph)))
                        else:
                            rand = ph
                        phoneme_str += rand
                    else:
                        phoneme_str += ph
                else:
                    # mask penuh fonem (pakai 'M')
                    phoneme_masked = self.token_mask * max(1, len(ph))
                    start = len(phoneme_str)
                    phoneme_str += phoneme_masked
                    masked_idx_list.extend(range(start, start + len(phoneme_masked)))
            else:
                phoneme_str += ph

            # tambahkan separator antar subtoken
            phoneme_str += self.token_separator
            labels_str += ph + self.token_separator
            words.append(self.word_separator)

        # potong jika terlalu panjang
        mel_length = len(phoneme_str)
        masked_index = []
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            end = random_start + self.max_mel_length
            phoneme_str = phoneme_str[random_start:end]
            labels_str = labels_str[random_start:end]
            for m in masked_idx_list:
                if random_start <= m < end:
                    masked_index.append(m - random_start)
        else:
            masked_index = masked_idx_list

        # ubah string → id (via TextCleaner)
        phoneme_ids = self.text_cleaner(phoneme_str)
        label_ids = self.text_cleaner(labels_str)

        phonemes_tensor = torch.LongTensor(phoneme_ids)
        labels_tensor = torch.LongTensor(label_ids)
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
        lengths = [b[0].shape[0] for b in batch]
        order = np.argsort(lengths)[::-1]
        batch = [batch[i] for i in order]

        max_seq_length = max(lengths)

        words = torch.full((batch_size, max_seq_length), self.text_pad_index, dtype=torch.long)
        labels = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        phonemes = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

        input_lengths, token_lengths, masked_indices = [], [], []

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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu')
    )
