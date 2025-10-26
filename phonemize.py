import re
import subprocess
import warnings
from functools import lru_cache
from transformers import AutoTokenizer
from text_normalize import normalize_text
from lingua import Language, LanguageDetectorBuilder

warnings.filterwarnings("ignore", message="Trying to detect language from a single word.")

languages = [Language.ENGLISH, Language.INDONESIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

@lru_cache(maxsize=100_000)
def detect_lang(word: str) -> str:
    result = detector.detect_language_of(word)
    if result is None:
        return "id"
    return "en" if result == Language.ENGLISH else "id"

@lru_cache(maxsize=100_000)
def phonemize_word(word: str, ipa: bool, keep_stress: bool, sep: str) -> str:
    lang = detect_lang(word)
    lang_map = {"id": "id", "en": "en-us"}
    voice = lang_map.get(lang, "id")
    cmd = ["espeak-ng", "-v", voice, "-q", f"--sep={sep}", word]
    if ipa:
        cmd.insert(3, "--ipa")
    else:
        cmd.insert(3, "-x")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        phonemes = result.stdout.decode("utf-8", errors="ignore").strip().replace("\ufeff", "")
        if not keep_stress:
            phonemes = re.sub(r"[ˈˌ]", "", phonemes)
        return phonemes or word
    except (subprocess.TimeoutExpired, Exception):
        return word

def phonemize(text, tokenizer):
    """Return BPE-aligned phonemization dict (flattened input_ids & 1-1 with phoneme_subtokens)."""
    original_text = text
    normalized_text = normalize_text(text)
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", normalized_text)
    detok_after = " ".join(words)

    all_input_ids = []
    all_tokens = []
    phoneme_words = []
    phoneme_subtokens = []

    for w in words:
        try:
            ids = tokenizer.encode(w, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            phon = phonemize_word(w, ipa=True, keep_stress=False, sep="")
        except Exception:
            continue
        if not ids:
            continue

        phoneme_words.append(w)
        all_input_ids.extend(ids)
        all_tokens.extend(tokens)
        phoneme_subtokens.extend([phon] * len(ids))

    return {
        "before": original_text,
        "after": detok_after,
        "phoneme_words": phoneme_words,
        "bpe_tokens": all_tokens,
        "input_ids": all_input_ids,          # flattened list[int]
        "phoneme_subtokens": phoneme_subtokens,  # len == len(input_ids)
    }

if __name__ == "__main__":
    sample_text = "Hello dunia saya belajar"
    tokenizer_name = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    result = phonemize(sample_text, tokenizer)
    print(len(result["input_ids"]), len(result["phoneme_subtokens"]))
