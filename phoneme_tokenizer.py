# coding: utf-8
import json


class PhonemeTokenizer:
    """
    Tokenizer untuk phoneme-level BERT.
    1 phoneme = 1 token ID.
    """

    def __init__(self, vocab=None,
                 pad_token="[PAD]",
                 mask_token="[MASK]",
                 unk_token="[UNK]"):

        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token

        if vocab is None:
            vocab = {}

        self.stoi = dict(vocab)                             # string → id
        self.itos = {i: s for s, i in vocab.items()}        # id → string

        # pastikan special tokens ada
        self._ensure_special_tokens()

    # ------------------------------------------------------------
    def _ensure_special_tokens(self):
        special_tokens = [self.pad_token, self.mask_token, self.unk_token]

        next_id = len(self.stoi)
        for tok in special_tokens:
            if tok not in self.stoi:
                self.stoi[tok] = next_id
                next_id += 1

        self.itos = {i: s for s, i in self.stoi.items()}

    # ------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path,
             pad_token="[PAD]",
             mask_token="[MASK]",
             unk_token="[UNK]"):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab,
                   pad_token=pad_token,
                   mask_token=mask_token,
                   unk_token=unk_token)

    # ------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------

    def convert_tokens_to_ids(self, token):
        return self.stoi.get(token, self.stoi[self.unk_token])

    def convert_ids_to_tokens(self, idx):
        return self.itos.get(idx, self.unk_token)

    def encode(self, phoneme_list):
        return [self.convert_tokens_to_ids(p) for p in phoneme_list]

    def decode(self, id_list):
        return [self.convert_ids_to_tokens(i) for i in id_list]

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------
    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_token_id(self):
        return self.stoi[self.pad_token]

    @property
    def mask_token_id(self):
        return self.stoi[self.mask_token]

    @property
    def unk_token_id(self):
        return self.stoi[self.unk_token]

    # ------------------------------------------------------------
    # Build vocab from dataset
    # ------------------------------------------------------------
    @classmethod
    def build_from_dataset(cls, dataset,
                           pad_token="[PAD]",
                           mask_token="[MASK]",
                           unk_token="[UNK]"):
        """
        dataset: list-of-dict
            {
               "phonemes": ["dˈua", "pˈuluh", ...],
               ...
            }
        """
        vocab = {}
        for item in dataset:
            for p in item["phonemes"]:
                if p not in vocab:
                    vocab[p] = len(vocab)

        return cls(vocab=vocab,
                   pad_token=pad_token,
                   mask_token=mask_token,
                   unk_token=unk_token)