# spacy_tokenizer.py
from __future__ import annotations

import pickle
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import spacy


def build_token_map(
    texts: Iterable[str],
    output_path: str | Path,
    *,
    spacy_model: str = "id_core_news_sm",
    fallback_blank: bool = True,
    lowercase: bool = True,
    min_freq: int = 1,
    special_tokens: Sequence[str] | None = None,
) -> Dict[str, int]:
    """Build a spaCy-based vocabulary and persist it to a pickle file."""
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "@@WORD_SEP@@"]

    output_path = Path(output_path)
    nlp = SpacyTokenizer._load_spacy(spacy_model, fallback_blank)

    counter: Counter[str] = Counter()
    for text in texts:
        if not text:
            continue
        doc = nlp(text)
        for token in doc:
            lexeme = token.text.strip()
            if not lexeme:
                continue
            if lowercase:
                lexeme = lexeme.lower()
            counter[lexeme] += 1

    token_to_id = OrderedDict()
    for token in special_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    if "<unk>" not in token_to_id:
        token_to_id["<unk>"] = len(token_to_id)

    with output_path.open("wb") as stream:
        pickle.dump({"token_to_id": dict(token_to_id)}, stream)

    return dict(token_to_id)


class SpacyTokenizer:
    """Tokenizer ringan berbasis spaCy dengan kompatibilitas BPE lama."""

    def __init__(
        self,
        vocab_path: str | Path,
        spacy_model: str = "id_core_news_sm",
        fallback_blank: bool = True,
        lowercase: bool = True,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.lowercase = lowercase
        self.nlp = self._load_spacy(spacy_model, fallback_blank)
        self.token_to_id = self._load_vocab(self.vocab_path)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.unk_token_id = self.token_to_id.get("<unk>")
        if self.unk_token_id is None:
            self.unk_token_id = max(self.token_to_id.values(), default=0) + 1
            self.token_to_id["<unk>"] = self.unk_token_id
            self.id_to_token[self.unk_token_id] = "<unk>"

    @staticmethod
    def _load_spacy(model: str, fallback_blank: bool):
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

    def decode(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "<unk>")

    def tokenize(self, text: str) -> List[str]:
        return [token.text for token in self.nlp(text)]

    def encode(self, token: str) -> List[int]:
        key = token.lower() if self.lowercase else token
        return [self.token_to_id.get(key, self.unk_token_id)]
