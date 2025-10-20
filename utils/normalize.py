import regex 
import unicodedata

PUNCT = regex.compile(r"(\p{P}|\p{Sm}|\p{Sc})")
HASHTAG_MENTION = regex.compile(r"(@\w+|#\w+)")
URL = regex.compile(r"https?://\S+|www\.\S+")

def normalize_symbols(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    protected = []
    def _protect(match):
        protected.append(match.group(0))
        return f"«{len(protected) - 1}»"

    text = URL.sub(_protect, text)
    text = HASHTAG_MENTION.sub(_protect, text)

    text = PUNCT.sub(r" \1 ", text)
    text = regex.sub(r"(?<=\p{N})%"," %", text)

    def _restore(match):
        return protected[int(match.group(1))]
    text = regex.sub(r"«(\d+)»", _restore, text)

    text = regex.sub(r"\s+", " ", text).strip()
    return text
