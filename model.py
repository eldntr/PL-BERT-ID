import torch
from torch import nn
import torch.nn.functional as F


class MultiTaskModel(nn.Module):
    """
    Multi-head model untuk PL-BERT v2:
      - token_head: prediksi masking/phoneme (CE loss)
      - word_head : prediksi subword/BPE (CTC loss)
    """
    def __init__(self, backbone, vocab_size_token=178, vocab_size_ctc=32000):
        super().__init__()
        self.encoder = backbone
        hidden_size = backbone.config.hidden_size

        # Head untuk token-level (phoneme / TextCleaner)
        self.token_head = nn.Linear(hidden_size, vocab_size_token)

        # Head untuk word-level (BPE / tokenizer)
        self.word_head = nn.Linear(hidden_size, vocab_size_ctc)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: tensor (B, T)
        attention_mask: tensor (B, T)
        return:
            tokens_pred: (B, T, vocab_size_token)
            words_pred : (B, T, vocab_size_ctc)
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        tokens_pred = self.token_head(hidden_states)
        words_pred = self.word_head(hidden_states)

        return tokens_pred, words_pred
