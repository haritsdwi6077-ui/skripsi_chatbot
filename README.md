# Skripsi RAG Chatbot — Hama & Penyakit Padi

Sistem tanya jawab berbasis *Retrieval Augmented Generation* (RAG) untuk hama dan
penyakit tanaman padi.

| Komponen | Teknologi |
|---|---|
| Antarmuka & API | Flask, di-*deploy* di Vercel |
| Vector database | Qdrant Cloud |
| Embedding | Jina AI (`jina-embeddings-v5-text-small`, 768 dimensi) |
| Generation (LLM) | Groq `openai/gpt-oss-120b` (endpoint kompatibel OpenAI, mudah diganti) |
| Evaluasi | RAGAS |

**Mengapa embedding memakai Jina AI?** Penyedia LLM (Groq, DeepSeek) hanya
menyediakan endpoint *chat completions*, bukan endpoint *embedding*. Jadi LLM
berperan pada tahap *generation*, sedangkan tahap *embedding* memakai Jina AI.

## Perubahan dari versi sebelumnya

| Sebelumnya | Sekarang | Alasan |
|---|---|---|
| Streamlit (`app.py`) | Flask + `templates/index.html` | sesuai proposal dan bisa di-*deploy* ke Vercel |
| `QdrantClient(":memory:")` | Qdrant Cloud | data tidak hilang dan bisa diakses aplikasi daring |
| Embedding ulang tiap kali dijalankan | sekali lewat `scripts/ingest.py` | hemat kuota API dan mempercepat *startup* |
| Rantai LangChain + penyedia embedding | pemanggilan API langsung | dependensi ringan, model mudah diganti |
| Dimensi vektor 3072 | 768 (dinormalisasi) | hemat penyimpanan Qdrant, pencarian lebih cepat |
| — | `scripts/evaluate.py` | metrik RAGAS untuk mode LLM + RAG dan LLM saja |

## Struktur

```
app.py                     Flask: halaman chat, /api/chat, /api/health
main.py                    menjalankan versi CLI
rag_core/config.py         konfigurasi dari environment variable
rag_core/embeddings.py     embedding dokumen & kueri (Jina AI)
rag_core/llm.py            pemanggilan LLM (format OpenAI, penyedia mudah diganti)
rag_core/engine.py         retrieval + prompt + generation, serta mode tanpa RAG
database/vector_db.py      koneksi & operasi Qdrant Cloud
interface/cli.py           antarmuka baris perintah
scripts/ingest.py          dijalankan lokal -> Qdrant Cloud
scripts/evaluate.py        dijalankan lokal -> skor RAGAS
data_source/               dokumen sumber & testset (tidak ikut di-deploy)
```

## Persiapan

1. Siapkan API key: penyedia LLM (misalnya Groq), Jina AI (https://jina.ai/embeddings), dan Qdrant Cloud.
2. Salin `.env.example` menjadi `.env`, lalu isi nilainya.
3. Pasang dependensi kerja lokal:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

4. Pastikan `LLM_MODEL` berisi nama model yang masih aktif menurut dokumentasi
   penyedia LLM. Nama model berubah dari waktu ke waktu, karena itu nilainya
   dibaca dari environment variable, bukan ditulis di dalam kode.

### Mengganti LLM

Cukup ubah `LLM_API_KEY`, `LLM_BASE_URL`, dan `LLM_MODEL` di `.env` (serta di
Vercel). Kosongkan `LLM_REASONING_EFFORT` untuk model non-reasoning. Embedding
dan Qdrant tidak terpengaruh, jadi tidak perlu ingest ulang.

## Membangun basis pengetahuan

Dokumen sumber ada di `data_source/`. Jalankan sekali:

```bash
pip install pymupdf          # opsional, membaca PDF jauh lebih cepat
python scripts/ingest.py --dry-run                                  # cek jumlah chunk, tanpa API
python scripts/ingest.py --recreate --batch 100 --paralel 3 --jeda 5
```

- Teks PDF disimpan di `.cache/teks/` dan embedding di `.cache/`, jadi menjalankan
  ulang tidak membaca PDF atau memanggil API lagi untuk data yang sama.
- Bila sering muncul `Jina 429`, turunkan `--paralel` atau naikkan `--jeda`.
- PDF hasil scan dilaporkan di awal proses. Ubah dulu menjadi teks (OCR) atau buang.

### Retrieval beragam (MMR)

Korpus memuat banyak artikel tentang blas, sehingga pada pertanyaan umum
("penyakit padi apa saja?") 4 potongan termirip bisa semuanya tentang blas.
Karena itu retrieval mengambil 40 kandidat (`RETRIEVAL_KANDIDAT`), lalu memilih
`TOP_K` potongan dengan *Maximal Marginal Relevance*: relevan dengan pertanyaan
tetapi tidak saling mirip (`MMR_LAMBDA=0.6`), maksimal 2 potongan per dokumen
(`MAKS_PER_SUMBER`). Matikan dengan `MMR_AKTIF=0`.

Susun dokumen per topik di subfolder, misalnya `data_source/Blas/`,
`data_source/HDB/`, `data_source/Tungro/`. Nama subfolder dipakai sebagai topik:
kandidat yang skornya berdekatan (`JENDELA_SKOR=0.05`) lebih dulu diambil satu per
topik, sehingga pertanyaan umum mendapat beberapa topik sekaligus, sedangkan
pertanyaan spesifik tetap spesifik. Setelah memindahkan file ke subfolder,
jalankan `python scripts/ingest.py --recreate`; kuota Jina tidak terpakai karena
cache embedding berdasarkan isi teks.

Pertanyaan bertipe daftar ("apa saja", "sebutkan", "jenis-jenis", "macam-macam")
mendapat `TOP_K_DAFTAR=8` potongan, pertanyaan lain `TOP_K=4`. Untuk pertanyaan
daftar, sistem juga mencari 2 potongan terbaik di setiap topik (filter field
`topik` di Qdrant), lalu mengambil satu potongan per topik. Dengan begitu topik
yang kalah jumlah dokumen, misalnya tungro, tetap muncul.

Periksa potongan yang diambil untuk sebuah pertanyaan:

```bash
python scripts/cek_retrieval.py "apa saja penyakit utama pada padi?"
```

Model embedding dan `EMBEDDING_DIM` harus sama antara ingestion dan aplikasi.
Jika salah satunya diubah, jalankan ulang dengan `--recreate`.

## Menjalankan

```bash
python main.py                  # versi CLI, mode RAG
python main.py --tanpa-rag      # versi CLI, LLM saja (pembanding)
flask --app app run --debug     # versi web, buka http://127.0.0.1:5000
```

Percakapan bersifat berkelanjutan: pertanyaan lanjutan seperti "cara
pencegahannya gimana?" ditulis ulang oleh LLM menjadi pertanyaan mandiri
(misalnya "Bagaimana cara pencegahan beluk pada tanaman padi?") sebelum
pencarian dokumen. Riwayat yang dipakai maksimal 3 giliran terakhir. Ketik
`baru` di CLI atau tekan **Percakapan baru** di web untuk memulai ulang.
Evaluasi tetap satu pertanyaan per uji, tanpa riwayat.

Halaman web punya pilihan **Mode jawaban**: *Dengan RAG* (bawaan) dan *Tanpa RAG*
(LLM saja, pembanding). Mode tanpa RAG juga bisa dibuka langsung lewat
`http://127.0.0.1:5000/?mode=llm`. Untuk versi yang dipakai petani, sembunyikan
pilihan ini dengan environment variable `TAMPILKAN_MODE=0`.

## Deploy ke Vercel

1. Push repo ini ke GitHub, lalu *import* di Vercel.
2. Isi seluruh variabel `.env` pada **Settings → Environment Variables**.
   Jangan pernah meng-*commit* file `.env`.
3. Setelah deploy, cek `/api/health` untuk memastikan konfigurasi terbaca.

Vercel hanya menjalankan `app.py`. Folder `data_source/`, `scripts/`, dan
`interface/` dikecualikan lewat `vercel.json` agar ukuran *bundle* tetap kecil.

## Evaluasi RAGAS

Susun `data_source/testset.json` (contoh: `data_source/testset.example.json`)
berisi 50–100 pasang pertanyaan dan jawaban rujukan, lalu:

Isi dulu `JUDGE_MODEL`, `JUDGE_BASE_URL`, dan `JUDGE_API_KEY` di `.env` dengan model
yang berbeda dari model yang diuji. `evaluate.py` berhenti bila penilainya sama.

```bash
python scripts/evaluate.py --mode rag --out coba_rag.csv --batas 3   # uji coba 3 soal
python scripts/evaluate.py --mode rag --out hasil_rag.csv   # LLM + RAG
python scripts/evaluate.py --mode llm --out hasil_llm.csv   # LLM saja
python scripts/rekap_hasil.py --rag hasil_rag.csv --llm hasil_llm.csv   # tabel BAB IV
```

`rekap_hasil.py` menghitung rata-rata, simpangan baku, minimum, dan maksimum setiap
metrik, uji Wilcoxon berpasangan pada answer relevancy dan factual correctness,
waktu respons, serta rata-rata per jenis pertanyaan, lalu menyimpannya ke
`rekap_hasil.md`. Skor kosong (NaN) tidak dihitung sebagai nol dan didaftar di akhir.
Opsi `--jeda` (default 2 detik) dan `--workers` (default 4) pada `evaluate.py` dapat
diatur bila sering terkena batas kuota API.

| Metrik | `--mode rag` | `--mode llm` |
|---|---|---|
| Context precision | ya | tidak (tidak ada konteks) |
| Context recall | ya | tidak |
| Faithfulness | ya | tidak |
| Answer relevancy | ya | ya |
| Factual correctness | ya | ya |

Perbandingan LLM + RAG dengan LLM saja memakai metrik yang ada di kedua mode
(answer relevancy dan factual correctness), serta waktu respons. Gunakan testset,
LLM, dan LLM penilai yang sama untuk kedua mode.

Untuk membandingkan beberapa LLM, jalankan perintah yang sama dengan `--model`
berbeda tanpa mengubah komponen lain, agar selisih skor benar-benar berasal dari
LLM-nya. Gunakan LLM penilai yang sama untuk semua model yang diuji lewat
`JUDGE_MODEL`, `JUDGE_BASE_URL`, dan `JUDGE_API_KEY`.
