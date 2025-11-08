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
            'phoneme': phoneme_tensor,
            'target_phoneme': target_phoneme_tensor,
            'bpe_target': bpe_tensor,
            'input_length': len(phoneme_tensor),
            'taget_length': len(bpe_tensor),
            'masked_indices': masked_idx
        }
        
class Collater(object):
    def __call__(self, batch):
        batch_size = len(batch)
        max_input_len = max([b['input_length'] for b in batch])
        max_text_length = max([b['taget_length'] for b in batch])
        
        phonemes = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        target_phonemes = torch.zeros_like(phonemes)
        bpe_targets = torch.zeros((batch_size, max_text_length), dtype=torch.long)
        input_lengths, target_lengths = [], []
        masked_indices = []
        
        for i, b in enumerate(batch):
            l_in, l_out = b['input_length'], b['taget_length']
            phonemes[i, :l_in] = b['phoneme']
            target_phonemes[i, :l_in] = b['target_phoneme']
            bpe_targets[i, :l_out] = b['bpe_target']
            input_lengths.append(l_in)
            target_lengths.append(l_out)
            masked_indices.append(b['masked_indices'])
        
        return phonemes, target_phonemes, bpe_targets, input_lengths, target_lengths, masked_indices

def build_dataloader(dataset,
                     batch_size=4,
                     num_workers=0,
                     validation=False,
                     device='cpu'):

    collate_fn = Collater()
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=not validation,
                      collate_fn=collate_fn,
                      num_workers=num_workers,
                      pin_memory=(device != 'cpu'))
