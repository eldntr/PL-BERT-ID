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
from transformers import TransfoXLTokenizer

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

# Di dalam file phonemize.py

def phonemize(text, global_phonemizer, tokenizer):
    """
    Fungsi yang telah disesuaikan untuk menangani subword tokenization dengan benar.
    """
    # 1. Pisahkan teks menjadi kata-kata utuh berdasarkan spasi.
    # Ini adalah langkah krusial yang sebelumnya tidak ada.
    original_words = text.strip().split()
    
    input_ids = []
    phonemes = []
    
    # 2. Iterasi melalui setiap KATA UTUH.
    for word in original_words:
        # Jika kata adalah tanda baca, perlakukan secara terpisah.
        if word in string.punctuation:
            # Tokenisasi tanda baca
            word_ids = tokenizer.encode(word)
            if not word_ids: continue # Lanjut jika tokenizer mengabaikan karakter ini
            
            input_ids.append(word_ids[0])
            phonemes.append(word) # Tanda baca tidak memiliki fonem, jadi gunakan karakter aslinya
            continue

        # 3. Fonemisasi kata utuh untuk mendapatkan pengucapan yang benar.
        phoneme_str = phonemize_word(word, True, True, "")
        
        # 4. Tokenisasi kata utuh yang sama untuk mendapatkan ID subword.
        word_ids = tokenizer.encode(word)
        
        if not word_ids:
            continue
            
        # 5. Petakan hasil fonem ke subword.
        # Fonem hanya untuk token pertama, sisanya diberi label kosong.
        for i, token_id in enumerate(word_ids):
            input_ids.append(token_id)
            if i == 0:
                phonemes.append(phoneme_str)
            else:
                phonemes.append("") # atau bisa menggunakan penanda lain seperti "<cont>"

    assert len(input_ids) == len(phonemes)
    return {'input_ids': input_ids, 'phonemes': phonemes}

def main():
    # Test Indonesian & English Phonemizer
    print("=" * 60)
    print("TEST INDONESIAN & ENGLISH PHONEMIZER")
    print("=" * 60)
    
    import os
    os.environ['TRUST_REMOTE_CODE'] = 'True'
    
    # Initialize phonemizer
    phonemizer_instance = EnIndPhonemizer(ipa=True, keep_stress=True)
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    
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
