import string
from transformers import AutoTokenizer
from phonemize_data import EnIndPhonemizer


def chunk_sequence(seq, size=512, overlap=64):
    if len(seq) <= size:
        return [seq]
    chunks = []
    step = size - overlap
    for i in range(0, len(seq), step):
        chunks.append(seq[i:i + size])
        if i + size >= len(seq):
            break
    return chunks


def phonemize(text, global_phonemizer, tokenizer):
    text = text.strip().lower()

    # phoneme-ize per kata
    words = text.split()
    phonemes_word = [global_phonemizer.phonemize(w) for w in words]
    joined = " ".join(phonemes_word).replace("  ", " ").strip()
    phoneme_seq = list(joined)

    # tokenisasi tanpa limit 2048
    bpe_tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)

    # chunking aman
    ph_chunks = chunk_sequence(phoneme_seq, size=512, overlap=64)
    id_chunks = chunk_sequence(input_ids, size=256, overlap=32)
    tok_chunks = chunk_sequence(bpe_tokens, size=256, overlap=32)

    N = min(len(ph_chunks), len(id_chunks))
    return [
        {
            "phonemes": ph_chunks[i],
            "input_ids": id_chunks[i],
            "bpe_tokens": tok_chunks[i],
            "chunk_id": i,
        }
        for i in range(N)
    ]


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    tok.model_max_length = int(1e9)
    tok.model_input_names = ["input_ids"]
    phn = EnIndPhonemizer(ipa=True, keep_stress=False)
    r = phonemize("Ini contoh teks panjang untuk diuji.", phn, tok)
    print(len(r), r[0])
