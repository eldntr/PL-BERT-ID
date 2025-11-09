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
from torch.utils.data import DataLoader, Dataset

from text_utils import TextCleaner

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

MAX_PHONEME_LEN = 512
MAX_BPE_LEN = 256

class FilePathDataset(Dataset):
    def __init__(self, dataset,
                 token_maps="token_maps.pkl",
                 word_mask_prob=0.15,
                 replace_prob=0.2,
                 phoneme_mask_prob=0.1,
                 token_mask="[M]"):
        """
        dataset: hasil phonemize (list of dict {phonemes, input_ids, bpe_tokens})
        token_maps: mapping fonem ke ID numerik
        """
        with open(token_maps, 'rb') as handle:
            self.token_maps = pickle.load(handle)                 
        
        self.dataset = dataset
        self.word_mask_prob = word_mask_prob
        self.replace_prob = replace_prob        
        self.phoneme_mask_prob = phoneme_mask_prob
        self.token_mask = token_mask 
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        phonemes = entry['phonemes']
        bpe_ids = entry['input_ids']

        phoneme_ids = []
        for p in phonemes:
            if p in self.token_maps:
                phoneme_ids.append(self.token_maps[p]['token'])
            else:
                phoneme_ids.append(self.token_maps.get('<unk>', {'token': 0})['token'])
                
        masked_idx = []
        masked_phoneme_ids = phoneme_ids.copy()
        for i in range(len(masked_phoneme_ids)):
            if np.random.rand() < self.word_mask_prob:
                r = np.random.rand()
                if r < 0.8:
                    masked_phoneme_ids[i] = self.token_maps.get(self.token_mask, {'token': 0})['token']
                elif r < 0.9:
                    masked_phoneme_ids[i] = random.choice(list(self.token_maps.values()))['token']
                masked_idx.append(i)
        
        phoneme_tensor = torch.LongTensor(masked_phoneme_ids)
        target_phoneme_tensor = torch.LongTensor(phoneme_ids)
        bpe_tensor = torch.LongTensor(bpe_ids)
       
        return {
            'phonemes': phoneme_tensor,
            'target_phonemes': target_phoneme_tensor,
            'bpe_targets': bpe_tensor,
            'input_length': len(phoneme_tensor),
            'target_length': len(bpe_tensor),
            'masked_indices': masked_idx
        }
        
MAX_PHONEME_LEN = 512
MAX_BPE_LEN = 256

class Collater:
    def __call__(self, batch):
        # Crop per-sample dulu agar tidak ada yang lewat limit
        cropped = []
        for b in batch:
            # nama key bisa sesuaikan milik kamu; contoh umum:
            ph = b["phonemes"][:MAX_PHONEME_LEN]
            tph = b["target_phonemes"][:MAX_PHONEME_LEN] if "target_phonemes" in b else b["phonemes"][:MAX_PHONEME_LEN]
            bp = b["bpe_targets"][:MAX_BPE_LEN]
            cropped.append({
                "phonemes": ph,
                "target_phonemes": tph,
                "bpe_targets": bp,
                "input_length": len(ph),
                "target_length": len(bp),
                "masked_indices": [i for i in b["masked_indices"] if i < len(ph)]
            })

        batch = cropped

        B = len(batch)
        max_in = max(b["input_length"] for b in batch)
        max_out = max(b["target_length"] for b in batch)

        phonemes = torch.zeros((B, max_in), dtype=torch.long)
        target_phonemes = torch.zeros_like(phonemes)
        bpe_targets = torch.zeros((B, max_out), dtype=torch.long)

        input_lengths, target_lengths, masked_indices = [], [], []

        for i, b in enumerate(batch):
            l_in, l_out = b["input_length"], b["target_length"]
            phonemes[i, :l_in] = torch.as_tensor(b["phonemes"], dtype=torch.long)
            target_phonemes[i, :l_in] = torch.as_tensor(b["target_phonemes"], dtype=torch.long)
            bpe_targets[i, :l_out] = torch.as_tensor(b["bpe_targets"], dtype=torch.long)
            input_lengths.append(l_in)
            target_lengths.append(l_out)
            masked_indices.append(b["masked_indices"])

        return phonemes, target_phonemes, bpe_targets, input_lengths, target_lengths, masked_indices


def build_dataloader(dataset, batch_size=4, num_workers=0, validation=False, device='cpu'):
    collate_fn = Collater()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu')
    )
