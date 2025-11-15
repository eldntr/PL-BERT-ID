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
from torch.utils.data import Dataset, DataLoader

from text_utils import TextCleaner
from phoneme_tokenizer import PhonemeTokenizer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)


def pad_list(seq, max_len, pad_value):
    return seq + [pad_value] * (max_len - len(seq))


def pad_2d(list_of_seq, pad_value):
    max_len = max(len(seq) for seq in list_of_seq)
    return [pad_list(seq, max_len, pad_value) for seq in list_of_seq]


class FilePathDataset(Dataset):
    """
    Dataset untuk PL-BERT CTC:
    - Input: phoneme sequence
    - MLM target: phoneme (masking)
    - CTC targer: flatten BPE sequence
    """
    def __init__(self, 
                 dataset,
                 phoneme_tokenizer,
                 bpe_tokenizer,
                 mlm_ratio=0.15):
        """
        dataset: list-of-dict
            {
                "phonemes": ["həlˈoʊ", "dˈua", ...]
                "input_ids": [[15339], [1072, 64], ...]
            }
        """
        self.data = dataset
        self.ph_tok = phoneme_tokenizer
        self.bpe_tok = bpe_tokenizer
        self.mlm_ratio = mlm_ratio
        
        # pad values
        self.pad_ph = phoneme_tokenizer.pad_token_id
        self.mask_id = phoneme_tokenizer.mask_token_id
        self.pad_bpe = bpe_tokenizer.pad_token_id
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Phoneme sequenxe
        phonemes = self.data[idx]['phonemes']
        X_ph = self.ph_tok.encode(phonemes)
        T_ph = len(X_ph)
        
        # Flatten BPE sequence for CTC
        flat_bpe = []
        for seq in item["input_ids"]:
            flat_bpe.extend(seq)
        T_bpe = len(flat_bpe)
        
        # Whole-phoneme masking (15%)
        num_mask = max(1, int(T_ph * self.mlm_ratio))
        masked_indices = random.sample(range(T_ph), num_mask)
        
        X_masked = X_ph.copy()
        
        # MLM labels: hanya di posisi masked
        ignore_index = -100
        Y_mlm = [ignore_index] * T_ph
        
        for mi in masked_indices:
            rnd = random.random()
            if rnd < 0.8:
                X_masked[mi] = self.mask_id
            elif rnd < 0.9:
                X_masked[mi] = random.randint(0, self.ph_tok.vocab_size - 1)
            else:
                X_masked[mi] = X_ph[mi]
                
            # target MLM di posisi ini = token asli
            Y_mlm[mi] = X_ph[mi]
        
        return {
            "X_ph": X_masked,
            "Y_mlm": Y_mlm,         # -100 untuk label non-masked
            "Y_p2g": flat_bpe,
            "T_ph": T_ph,
            "T_bpe": T_bpe
        }        
                
def collater(batch):
    """
    Output: 
        X_ph : (B, max_T_ph)
        Y_mlm: (B, max_T_ph)
        Y_p2g: (B, max_T_bpe)
        input_lengths : (B,)
        target_lengths: (B,)
    """
    X_list = [item["X_ph"] for item in batch]
    Y_mlm_list = [item["Y_mlm"] for item in batch]
    Y_ctc_list = [item["Y_p2g"] for item in batch]
    
    input_lengths = torch.tensor([item["T_ph"] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["T_bpe"] for item in batch], dtype=torch.long)

    # Padding
    pad_ph = 0
    pad_bpe = 0
    
    ignore_index = -100
    
    X_ph = torch.tensor(pad_2d(X_list, pad_ph), dtype=torch.long)
    Y_mlm = torch.tensor(pad_2d(Y_mlm_list, ignore_index), dtype=torch.long)
    Y_ctc = torch.tensor(pad_2d(Y_ctc_list, pad_bpe), dtype=torch.long)
    
    return {
        "X_ph": X_ph,                   # (B, max_T_ph)
        "Y_mlm": Y_mlm,                 # (B, max_T_ph)
        "Y_p2g": Y_ctc,                 # (B, max_T_bpe)
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }

def build_dataloader(dataset,
                     phoneme_tokenizer,
                     bpe_tokenizer,
                     batch_size=4,
                     shuffle=True,
                     num_workers=0):

    ds = FilePathDataset(
        dataset=dataset,
        phoneme_tokenizer=phoneme_tokenizer,
        bpe_tokenizer=bpe_tokenizer
    )
    
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collater
    )
    
    return dl