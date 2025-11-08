import string
# from text_normalize import normalize_text, remove_accents
from transformers import AutoTokenizer
from phonemize_data import EnIndPhonemizer

def phonemize(text, global_phonemizer, tokenizer):
    # text = normalize_text(remove_accents(text))
    text = text.strip().lower()
    
    bpe_tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    words = text.split()
    phonemes_word = [global_phonemizer.phonemize(w) for w in words]
    
    joined_phonemes = " ".join(phonemes_word)
    joined_phonemes = joined_phonemes.replace("  ", " ").strip()
    
    phonemes_sequence = list(joined_phonemes)
    
    return {
        "phonemes": phonemes_sequence,
        "input_ids": input_ids,
        "bpe_tokens": bpe_tokens
    }
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

    global_phonemizer = EnIndPhonemizer(ipa=True, keep_stress=False)

    sample_text = "Ini contoh teks untuk diuji."
    result = phonemize(sample_text, global_phonemizer, tokenizer)
    print(result)