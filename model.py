import torch
import torch.nn as nn

try:
    from transformers import AlbertModel, AlbertConfig
    HF_AVAILABLE = True
except ImportError:
    AlbertModel = None
    AlbertConfig = None
    HF_AVAILABLE = False
    

class MultiTaskModel(nn.Module):
    """
    Multi-task model:
    - Encoder: phoneme-level BERT (ALBERT)
    - Head 1 : MLM (predict phoneme IDs)
    - Head 2 : CTC (predict BPE IDs + blank) dengan upsampling time-step
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        phoneme_vocab_size: int,
        bpe_vocab_size: int,
        pad_token_id: int = 0,
        blank_id: int = 0,
        upsample_factor: int = 4,
    ):
        """
        Args
        ----
        encoder : nn.Module
            Encoder model, AlbertModel.
            Harus mengembalikan 'last_hidden_state' dengan shape (B, T, H).
        phoneme_vocab_size : int
            Ukuran vocab untuk phoneme (MLM head).
        bpe_vocab_size : int
            Ukuran vocab BPE (tanpa blank),
            Model otomatis menambah +1 untuk blank
        pad_token_id : int
            ID token PAD untuk phoneme (untuk attention_mask)
        blank_id : int
            Index untuk CTC blank.
            Default 0.
        upsample_factor : int
            Faktor upsampling untuk CTC output.
            Default 4.
        """
        super().__init__()
        
        self.encoder = encoder
        self.pad_token_id = pad_token_id
        self.blank_id = blank_id
        self.upsample_factor = upsample_factor

        # Ambil hidden_size dari config, tidak ada fallback jadi pastikan ada
        hidden_size = encoder.config.hidden_size
        
        # MLM head: hidden_size -> phoneme_vocab_size
        self.mlm_head = nn.Linear(hidden_size, phoneme_vocab_size)
        
        # CTC head: hidden_size -> (bpe_vocab_size + 1) (+1 untuk blank)
        self.ctc_output_dim = bpe_vocab_size + 1
        self.ctc_head = nn.Linear(hidden_size, self.ctc_output_dim)
        
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        """
        Forward pass.
        
        Args
        ----
        input_ids : LongTensor
            Shape: (B, T_phoneme)
        attention_mask : LongTensor or None
            Shape: (B, T_phoneme). Jika None, otomatis dibuat dari pad_token_id.
        output_hidden_states : bool
            Jika True, akan mengembalikan hidden_states dari encoder.
            
        Returns
        -------
        logits_mlm : FloatTensor
            Shape: (B, T_phoneme, phoneme_vocab_size)
        logits_ctc : FloatTensor
            Shape: (B, T_phoneme * upsample_factor, bpe_vocab_size + 1)
        extra : dict
            extra["encoder_outputs"] = output dari encoder (debug, opsional)
        """
        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = (input_ids != self.pad_token_id).long()
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        hidden = encoder_outputs.last_hidden_state  # (B, T_ph, H)
        
        # MLM head di level phoneme
        logits_mlm = self.mlm_head(hidden)          # (B, T_ph, phoneme_vocab_size)
        
        # Upsampling untuk CTC head
        # hidden: (B, T_ph, H) -> (B, T_ph * upsample_factor, H)
        hidden_ctc = hidden.repeat_interleave(self.upsample_factor, dim=1)
        
        # CTC head (phoneme -> BPE + blank)
        logits_ctc = self.ctc_head(hidden_ctc)      # (B, T_up, bpe_vocab_size+1)
        
        extra = {
            "encoder_outputs": encoder_outputs,
            "hidden_ctc": hidden_ctc,
        }
        return logits_mlm, logits_ctc, extra
    
    def get_ctc_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Helper untuk menghitung panjang input CTC setelah upsampling.
        
        Args
        ----
        input_lengths : torch.Tensor
            Panjang T_ph per sample (before upsampling).
            
        Returns
        -------
        torch.Tensor
            Panjang T_ph * upsample_factor per sample.
        """
        return input_lengths * self.upsample_factor
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
    
    @classmethod
    def from_albert_config(
        cls,
        phoneme_vocab_size: int,
        bpe_vocab_size: int,
        albert_config_dict: dict,
        pad_token_id: int = 0,
        blank_id: int = 0,
        upsample_factor: int = 4,
    ):
        """
        Helper classmethod untuk membuat ALBERT 
        """
        
        cfg = AlbertConfig(**albert_config_dict)
        cfg.vocab_size = phoneme_vocab_size
        
        encoder = AlbertModel(cfg)
        model = cls(
            encoder=encoder,
            phoneme_vocab_size=phoneme_vocab_size,
            bpe_vocab_size=bpe_vocab_size,
            pad_token_id=pad_token_id,
            blank_id=blank_id,
            upsample_factor=upsample_factor,
        )
        return model