# model_ctc_bpe.py
import torch
from torch import nn

class MultiTaskModel(nn.Module):
    def __init__(self, model, num_tokens=178, num_vocab=84827, hidden_size=768, dropout=0.1):
        """
        model: backbone encoder (mis. AlbertModel)
        num_tokens: jumlah vocab phoneme (dari token_maps.pkl)
        num_vocab: jumlah vocab BPE + 1 (untuk blank)
        hidden_size: dimensi hidden encoder
        """
        super().__init__()

        self.encoder = model

        # Head untuk MLM phoneme (masked token prediction)
        self.mask_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tokens)
        )

        # Head untuk CTC (BPE sequence prediction)
        self.word_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_vocab)  # vocab_size + 1, index 0 reserved for blank
        )

    def forward(self, phonemes, attention_mask=None):
        """
        phonemes: tensor [B, T]
        attention_mask: tensor [B, T]
        return:
            tokens_pred: [B, T, num_tokens]
            words_pred : [B, T, num_vocab]
        """
        outputs = self.encoder(phonemes, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state  # [B, T, H]
        tokens_pred = self.mask_predictor(hidden_states)
        words_pred = self.word_predictor(hidden_states)

        return tokens_pred, words_pred
