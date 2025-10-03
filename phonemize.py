import string
import subprocess
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
import warnings
from tqdm import tqdm
from spacy_tokenizer import SpacyTokenizer

warnings.filterwarnings("ignore", message="Trying to detect language from a single word.")

languages = [Language.ENGLISH, Language.INDONESIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()
DEFAULT_TOKEN_MAP_PATH = Path(__file__).with_name("token_maps.pkl")
_TOKENIZER: SpacyTokenizer | None = None


def get_default_tokenizer() -> SpacyTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = SpacyTokenizer(DEFAULT_TOKEN_MAP_PATH)
    return _TOKENIZER

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

class EnIndPhonemizer:
    def __init__(self, ipa=True, keep_stress=False, sep="", max_workers=None):
        self.ipa = ipa
        self.keep_stress = keep_stress
        self.sep = sep
        self.max_workers = max_workers or 4

    def phonemize(self, text: str) -> str:
        if not text:
            return ""
        words = text.strip().split()
        phonemized_words = [
            phonemize_word(w, self.ipa, self.keep_stress, self.sep) for w in words
        ]
        return " ".join(phonemized_words)

    def process_in_parallel(self, texts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(
                tqdm(
                    executor.map(self.phonemize, texts),
                    total=len(texts),
                    desc="Phonemizing Sentences"
                )
            )

def phonemize(text, global_phonemizer, tokenizer: SpacyTokenizer | None = None):
    """Fungsi kompatibilitas untuk kode lama"""
    tokenizer = tokenizer or get_default_tokenizer()
    words = tokenizer.tokenize(text)
    
    phonemes_bad = [phonemize_word(word, True, True, "") if word not in string.punctuation else word for word in words]
    input_ids = []
    phonemes = []
    
    for i in range(len(words)):
        word = words[i]
        phoneme = phonemes_bad[i]
        
        # # Hanya proses kasus khusus untuk "@" (email, mention)
        # if "@" in word and len(word) > 1: # remove "@"
        #     phonemes.append(word.replace('@', ''))
        #     input_ids.append(tokenizer.encode(word.replace('@', ''))[0])
        #     continue
        
        input_ids.append(tokenizer.encode(word)[0])
        phonemes.append(phoneme)
        
    assert len(input_ids) == len(phonemes)
    return {'input_ids' : input_ids, 'phonemes': phonemes}

def main():
    # Test Indonesian & English Phonemizer
    print("=" * 60)
    print("TEST INDONESIAN & ENGLISH PHONEMIZER")
    print("=" * 60)
    
    import os
    os.environ['TRUST_REMOTE_CODE'] = 'True'
    
    # Initialize phonemizer
    phonemizer_instance = EnIndPhonemizer(ipa=True, keep_stress=True)
    tokenizer = get_default_tokenizer()
    
    # Test sentences in Indonesian and English
    test_sentences = [
        "Muhammad Anwar el-Sadat; ) adalah seorang politikus Mesir yang menjabat sebagai presiden Mesir ketiga, dari 15 Oktober 1970 hingga pembunuhannya oleh perwira tentara fundamentalis pada 6 Oktober 1981.",
        "This is an example sentence in English.",
        "Muhammad Anwar el-Sadat adalah seorang politikus Mesir.",
        "He was the third President of Egypt.",
        "Pada tahun 1978, Sadat dan Menachem Begin menandatangani perjanjian damai.",
        "They were awarded the Nobel Peace Prize.",
        "Saya suka makan nasi goreng.",
        "I like to eat fried rice.",
        "Selamat pagi, apa kabar?",
        "Good morning, how are you?",
        "eldintaro@gmail.com"
    ]
    
    print("\nTesting sentences:")
    print("-" * 40)
    
    for sentence in test_sentences:
        phoneme_result = phonemize(sentence, phonemizer_instance, tokenizer)
        print(f"Text:    {sentence}")
        print(f"Input IDs: {phoneme_result['input_ids']}")
        print(f"Phonemes: {phoneme_result['phonemes']}")
        print()
    
    print("Test completed!")

if __name__ == "__main__":
    main()
