import re
import subprocess
import warnings
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer
from text_normalize import normalize_text

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
        phonemes = result.stdout.decode("utf-8", errors="ignore").strip()
        phonemes = phonemes.replace("\ufeff", "")
        if not keep_stress:
            phonemes = re.sub(r"[ˈˌ]", "", phonemes)
        return phonemes
    except (subprocess.TimeoutExpired, Exception):
        return word

def phonemize(text, tokenizer):
    original_text = text
    normalized_text = normalize_text(text)

    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", normalized_text)
    detok_after = " ".join(words)

    input_ids = []
    phonemes = []
    bpe_tokens = []
    decoded_tokens = []

    for w in words:
        try:
            ids = tokenizer.encode(w, add_special_tokens=False)
            phon = phonemize_word(w, True, True, "")
        except Exception as e:
            print(f"[WARN] Skip word '{w}' due to {e}")
            continue

        if len(ids) == 0:
            continue

        input_ids.append(ids)
        bpe_tokens.append(tokenizer.convert_ids_to_tokens(ids))
        decoded_tokens.append(tokenizer.decode(ids))
        phonemes.append(phon)

    assert len(input_ids) == len(phonemes), "Word vs phoneme mismatch"
    return {
        "before": original_text,
        "after": detok_after,
        "phoneme_words": words,
        "phonemes": phonemes,
        "input_ids": input_ids,
        "bpe_tokens": bpe_tokens,
        "decoded_tokens": decoded_tokens,
    }


if __name__ == "__main__":
    sample_text = "hello (dua puluh tiga januari dua ribu dua puluh dua belas sepuluh AM)"
    tokenizer_name = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        raise SystemExit(f"Gagal memuat tokenizer '{tokenizer_name}': {exc}") from exc

    result = phonemize(sample_text, tokenizer)

    print("\nPhonemized output:")
    for word, ids, phoneme in zip(result["phoneme_words"], result["input_ids"], result["phonemes"]):
        decoded = tokenizer.decode(ids).strip()
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"{word}\t{phoneme}\t{decoded}\t{tokens}")
