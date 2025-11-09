import string
# from text_normalize import normalize_text, remove_accents
from transformers import AutoTokenizer
from phonemize_data import EnIndPhonemizer

def chunk_sequence(seq, size=512, overlap=64):
    """Bagi urutan panjang jadi beberapa segmen ber-overlap."""
    if len(seq) <= size:
        return [seq]
    chunks = []
    step = size - overlap
    for i in range(0, len(seq), step):
        chunk = seq[i:i + size]
        chunks.append(chunk)
        if i + size >= len(seq):
            break
    return chunks

def phonemize(text, global_phonemizer, tokenizer):
    # text = normalize_text(remove_accents(text))
    text = text.strip().lower()
    
    bpe_tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    words = text.split()
    phonemes_word = [global_phonemizer.phonemize(w) for w in words]
    
    joined_phonemes = " ".join(phonemes_word)
    joined_phonemes = joined_phonemes.replace("  ", " ").strip()
    
    phoneme_sequence = list(joined_phonemes)
    phoneme_chunks = chunk_sequence(phoneme_sequence, size=512, overlap=64)
    bpe_chunks = chunk_sequence(input_ids, size=256, overlap=32)

    chunked_records = []
    for i, ph_chunk in enumerate(phoneme_chunks):
        bpe_chunk = bpe_chunks[i] if i < len(bpe_chunks) else []
        chunked_records.append({
            "phonemes": ph_chunk,
            "input_ids": bpe_chunk,
            "bpe_tokens": bpe_tokens[i:i+len(bpe_chunk)],
            "chunk_id": i
        })
    return chunked_records
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    global_phonemizer = EnIndPhonemizer(ipa=True, keep_stress=False)

    sample_text = "Ini contoh teks untuk diuji."
    result = phonemize(sample_text, global_phonemizer, tokenizer)
    print(result)