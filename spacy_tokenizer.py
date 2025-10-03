# spacy_tokenizer.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import spacy


class SpacyTokenizer:
    """Tokenizer ringan berbasis spaCy dengan kompatibilitas BPE lama."""

    def __init__(
        self,
        vocab_path: str | Path,
        spacy_model: str = "id_core_news_sm",
        fallback_blank: bool = True,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.nlp = self._load_spacy(spacy_model, fallback_blank)
        self.token_to_id = self._load_vocab(self.vocab_path)
        self.unk_token_id = self.token_to_id.get("<unk>")
        if self.unk_token_id is None:
            self.unk_token_id = max(self.token_to_id.values(), default=0) + 1
            self.token_to_id["<unk>"] = self.unk_token_id

    def _load_spacy(self, model: str, fallback_blank: bool):
        try:
            return spacy.load(model)
        except OSError:
            if not fallback_blank:
                raise
            return spacy.blank("id")

    def _load_vocab(self, path: Path) -> Dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)

        if isinstance(payload, dict) and "token_to_id" in payload:
            return dict(payload["token_to_id"])

        token_to_id: Dict[str, int] = {}
        for value in payload.values():
            if isinstance(value, dict) and {"word", "token"} <= value.keys():
                token_to_id[value["word"]] = value["token"]
                continue
            raise ValueError("Unsupported token_maps format detected.")

        return token_to_id

    def tokenize(self, text: str) -> List[str]:
        return [token.text for token in self.nlp(text)]

    def encode(self, token: str) -> List[int]:
        return [self.token_to_id.get(token, self.unk_token_id)]
