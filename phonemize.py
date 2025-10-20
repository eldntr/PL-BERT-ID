import string
import subprocess
import re
from functools import lru_cache
from lingua import Language, LanguageDetectorBuilder
import warnings
import unicodedata
from nltk.tokenize import TweetTokenizer

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
import re
import string
from nltk.tokenize import wordpunct_tokenize

def normalize_text(text):
    # Bersihkan karakter aneh tapi TIDAK hapus spasi
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # normalize multiple spaces
    text = re.sub(r"[^a-zA-Z0-9\s.,?!'\"-]", "", text)  # keep only readable chars
    return text

def phonemize(text, tokenizer):
    # Normalisasi ringan
    text = normalize_text(text)

    # Tokenisasi kata (lebih kuat dari TweetTokenizer untuk kalimat panjang)
    words = wordpunct_tokenize(text)
    words = [w for w in words if w not in string.punctuation and w.strip() != ""]

    input_ids = []
    phonemes = []

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
        phonemes.append(phon)

    # Debug
    print(f"Tokenized {len(words)} words → {len(input_ids)} encoded")
    for i, (w, ids) in enumerate(zip(words[:10], input_ids[:10])):
        toks = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {i+1:>2}. {w} → {toks}")

    assert len(input_ids) == len(phonemes), "Word vs phoneme mismatch"
    return {"input_ids": input_ids, "phonemes": phonemes}
