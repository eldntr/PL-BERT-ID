# FilePathDataset, Collater, dan `build_dataloader`

Dokumen ini memberikan penjelasan teknis mendalam mengenai pipeline pemuatan data pada `dataloader.py`. Seluruh proses mengubah data hasil pra-proses menjadi batch tensor siap latih untuk PL-BERT berbasis subword (BPE).

---

## Format Data Masukan

Setiap entri dataset (`dataset[idx]`) adalah `dict` dengan struktur:

| Kunci | Tipe | Contoh | Catatan |
| --- | --- | --- | --- |
| `phonemes` | `List[str]` | `["sˌaja", "ˈandʒiŋ"]` | Fonem per kata, hasil dari modul phonemizer. |
| `input_ids` | `List[List[int]]` | `[[128000, 82, 12874], [53191, 287]]` | Token BPE per kata, hasil `AutoTokenizer`. |

Data ini dihasilkan oleh `preprocess.py` yang melakukan normalisasi teks, mendeteksi bahasa, mengubah ke fonem, dan memetakan kata ke token BPE.

---

## FilePathDataset

### Inisialisasi dan Atribut

Ketika `FilePathDataset` dibuat, langkah berikut terjadi:

1. Simpan daftar sample mentah (`self.data = dataset`).
2. Inisialisasi parameter masking (`word_mask_prob`, `phoneme_mask_prob`, `replace_prob`) dan batas panjang (`max_mel_length`).
3. Buat instance `TextCleaner` yang memetakan karakter fonem ke indeks integer.
4. Inisialisasi tokenizer melalui `AutoTokenizer.from_pretrained(tokenizer)`.
5. Tentukan ID khusus:
   - `self.word_separator = tokenizer.eos_token_id` (jika tersedia) untuk memisahkan kata di tensor token.
   - `self.pad_id` mengambil `pad_token_id` jika ada, jika tidak fallback ke `word_separator` atau `0`.

Reproducibility dijaga dengan `np.random.seed(1)` dan `random.seed(1)` pada modul.

### Tanda Tangan Output

Pemanggilan `dataset[idx]` mengembalikan:

```
phonemes_tensor: torch.LongTensor [mel_length]
words_tensor   : torch.LongTensor [num_word_tokens]
labels_tensor  : torch.LongTensor [mel_length]
masked_index   : List[int]
```

- `phonemes_tensor` berisi phoneme yang telah dimasker (karakter → ID).
- `labels_tensor` berisi phoneme asli (sebelum masking).
- `words_tensor` merupakan flatten dari BPE token per kata ditambah `word_separator` setelah setiap kata (jika ada).
- `masked_index` menyimpan posisi karakter fonem yang dimasker setelah pemotongan (jika terjadi).

### Detail Langkah `__getitem__`

1. **Ambil sample**  
   ```python
   item = self.data[idx]
   phonemes = item["phonemes"]
   input_ids = item["input_ids"]
   ```
   Diasumsikan panjang `phonemes` dan `input_ids` sama.

2. **Flatten token BPE**  
   - `words.extend(bpe_ids)` menambahkan sub-token ke list datar.  
   - Jika `word_separator` ada, tambahkan ID tersebut sebagai limiter antar kata.

3. **Bangun string fonem dan label**  
   - `labels` menyimpan fonem asli dengan spasi (`token_separator`) antar kata.  
   - `phoneme` diinisialisasi kosong dan akan berisi hasil masking.

4. **Masking per kata**  
   - Gunakan `np.random.rand()` untuk memutuskan apakah kata dimasker (`word_mask_prob`).  
   - Jika masking aktif:
     - Dengan probabilitas `replace_prob`, fonem diganti dengan fonem acak sepanjang kata.  
     - Sisanya diganti repetisi karakter `token_mask` (`"M"`).
     - Posisi karakter bertopeng dicatat di `masked_idx_list`.
   - Jika tidak dimasker, fonem asli ditambahkan apa adanya.
   - Setelah setiap kata, tambahkan `token_separator` (default `" "`).

   **Sumber fonem acak:**  
   `phoneme_list = ''.join(phonemes)` sehingga karakter diambil dari seluruh fonem sample untuk menjaga distribusi simbol.

5. **Pemotongan Dinamis (`max_mel_length`)**  
   - Hitung `mel_length = len(phoneme)`.  
   - Jika melebihi batas, pilih `random_start` secara seragam.  
   - Potong `phoneme` dan `labels` pada rentang `[random_start, random_start + max_mel_length)`.  
   - `masked_index` di-offset ulang agar selaras dengan rentang baru dan dipotong bila berada di luar.

6. **Normalisasi → Tensor**  
   - `TextCleaner` memetakan tiap karakter ke indeks numerik. Karakter tak dikenal diarahkan ke indeks `dicts['U']`.  
   - Konversi ke `torch.LongTensor`.

7. **Pengembalian**  
   Return tuple `(phonemes_tensor, words_tensor, labels_tensor, masked_index)` sesuai detail di atas.

### Edge Case Penting

- **Tokenizer tanpa `eos_token_id`:** `word_separator` menjadi `None`, pemisah kata tidak ditambahkan. `token_lengths` di `Collater` tetap valid karena hanya mem-filter terhadap `pad_id`.
- **Karakter fonem di luar kamus:** Ditangani oleh `TextCleaner` dengan fallback ke `U`.
- **Kata tanpa token BPE:** Jika list token kosong, BPE tidak menambah token. Pastikan pra-proses tidak menghasilkan kondisi ini.
- **Masking di akhir string:** `token_separator` masih ditambahkan, sehingga panjang fonem minimal bertambah satu karakter per kata. Pastikan `max_mel_length` memperhitungkan spasi tambahan ini.

---

## Collater

`Collater` mengemas sample per entri menjadi batch dengan padding dan metadata panjang. Ia menerima parameter opsional:

| Parameter | Default | Penjelasan |
| --- | --- | --- |
| `tokenizer` | `None` | Digunakan untuk mengambil `pad_token_id` dan `eos_token_id`. |
| `return_wave` | `False` | Placeholder untuk kompatibilitas; tidak dipakai di implementasi saat ini. |
| `debug` | `False` | Saat `True`, dapat dipakai untuk logging manual ketika mengembangkan. |

### Proses `__call__(batch)`

1. **Urutkan sample**  
   - Hitung `lengths = [phoneme.shape[0] for phoneme, _, _, _ in batch]`.  
   - Urutkan descending demi efisiensi padding dan mempermudah penggunaan PackedSequence.

2. **Alokasi Tensor Batch**  
   - `words`: ukuran `(batch_size, max_seq_length)`, diisi `pad_id`.  
   - `phonemes` dan `labels`: ukuran sama, diisi `0`.  
   - `masked_indices`: list kosong untuk setiap sample.

3. **Salin Data Per-Baris**  
   - `seq_len = min(phoneme.size(0), label.size(0), max_seq_length)`.  
   - `word_len = min(word.size(0), max_seq_length)`.  
   - Salin slice tensor ke batch arrays.  
   - Tambahkan `seq_len` ke `input_lengths`.

4. **Panjang Token Valid**  
   - `word_slice = words[bid, :word_len]`.  
   - `valid_mask = (word_slice != pad_id)` dan bila `word_separator` tidak `None`, filter tambahan `(word_slice != word_separator)`.  
   - `token_lengths` menyimpan jumlah token BPE yang akan dipakai oleh loss (menghindari PAD/EOS).

5. **Masked Indices**  
   - Setiap `masked_index` difilter agar `< seq_len` supaya tidak menunjuk ke padding.

6. **Pengembalian**  
   Tuple `(words, labels, phonemes, input_lengths, masked_indices, token_lengths)`.

### Properti Output

- Semua tensor bertipe `torch.long`.  
- `input_lengths` cocok dengan dimensi waktu untuk fitur akustik (fonem).  
- `token_lengths` memudahkan saat menghitung CTC atau mask attention karena menandai jumlah token non-pad.

### Pertimbangan Kinerja

- Sorting by length meminimalkan padding → mempercepat perhitungan.  
- Padding dilakukan di CPU, sehingga set `pin_memory=True` di `DataLoader` dapat mengurangi latensi ketika memindahkan ke GPU.

---

## `build_dataloader`

Signature:

```python
def build_dataloader(
    df,
    validation=False,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config=None,
    dataset_config=None,
):
    ...
```

### Parameter

| Parameter | Default | Penjelasan |
| --- | --- | --- |
| `df` | — | Dataset mentah (list of dicts, `Dataset` Hugging Face, dsb.) yang dapat diindeks. |
| `validation` | `False` | Menentukan apakah `shuffle` dimatikan dan `drop_last` dinonaktifkan. |
| `batch_size` | `4` | Ukuran batch. Poin penting: `drop_last=True` saat training agar bentuk batch konsisten. |
| `num_workers` | `1` | Jumlah worker untuk proses data paralel. Dengan dataset berat, naikkan angka ini. |
| `device` | `"cpu"` | Jika selain CPU, `pin_memory=True` untuk optimisasi transfer ke GPU. |
| `collate_config` | `{}` | Override parameter `Collater`, misal `{"debug": True}`. |
| `dataset_config` | `{}` | Override parameter `FilePathDataset`, misal `{"max_mel_length": 384}`. |

### Urutan Eksekusi

1. Buat `dataset = FilePathDataset(df, **dataset_config)`.  
2. Gunakan tokenizer dari dataset untuk menginisialisasi `Collater`.  
3. Bangun `DataLoader`:
   ```python
   loader = DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=not validation,
       num_workers=num_workers,
       drop_last=not validation,
       collate_fn=collate_fn,
       pin_memory=(device != "cpu"),
   )
   ```
4. Kembalikan `loader` untuk dipakai di loop training/validasi.

### Integrasi dengan `Configs/config.yml`

File konfigurasi biasanya menyediakan `dataset_params` dan `dataloader_params`. Contoh:

```yaml
dataset_params:
  tokenizer: GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct
  max_mel_length: 512
  word_mask_prob: 0.15
dataloader_params:
  batch_size: 16
  num_workers: 8
```

Gunakan `dataset_params` sebagai `dataset_config` dan `dataloader_params` untuk argumen `build_dataloader`.

---

## Ringkasan Alur End-to-End

1. **Pra-proses** menghasilkan struktur `phonemes` dan `input_ids` per kata.  
2. **FilePathDataset**:
   - Flatten token BPE dengan pemisah EOS.
   - Masking fonem secara acak dengan kontrol probabilitas.
   - Potong sample panjang dan konversi ke indeks integer.
3. **Collater**:
   - Urutkan sample berdasarkan panjang.
   - Lakukan padding dan hitung metadata panjang.
4. **build_dataloader** menggabungkan keduanya dalam `torch.utils.data.DataLoader`.

Pipeline ini menjaga keselarasan antara fonem dan token subword, memudahkan perhitungan loss berbasis CTC, serta mendukung kode-switching melalui tokenizer dan fonemizer multibahasa.
