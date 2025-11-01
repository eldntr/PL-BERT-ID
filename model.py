import torch
from torch import nn

class MultiTaskModel(nn.Module):
    def __init__(self, model, num_tokens=178, num_vocab=32001, hidden_size=768):
        """
        num_vocab = vocab_size_tokenizer + 1 (slot terakhir untuk CTC blank).
        num_tokens = jumlah kelas token-phoneme untuk head masking.
        """
        super().__init__()
        self.encoder = model
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)   # token-level head
        self.word_predictor = nn.Linear(hidden_size, num_vocab)     # CTC head

    def forward(self, phonemes, attention_mask=None):
        # Albert expects attention_mask: 1 = keep, 0 = pad
        output = self.encoder(phonemes, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        tokens_pred = self.mask_predictor(hidden)
        words_pred = self.word_predictor(hidden)
        return tokens_pred, words_pred
