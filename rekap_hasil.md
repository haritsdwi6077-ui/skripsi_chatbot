# Rekap hasil evaluasi

Berkas: hasil_rag.csv, hasil_llm.csv

Jumlah pertanyaan: mode RAG = 2, mode tanpa RAG = 2

### Tabel 4.18 Hasil evaluasi RAGAS mode LLM dengan RAG

| Metrik | n | Rata-rata | Simpangan baku | Minimum | Maksimum |
|---|---|---|---|---|---|
| Context precision | 1 | 1,0000 | - | 1,0000 | 1,0000 |
| Context recall | 1 | 1,0000 | - | 1,0000 | 1,0000 |
| Faithfulness | 1 | 0,7500 | - | 0,7500 | 0,7500 |
| Answer relevancy | 0 | - | - | - | - |
| Factual correctness | 2 | 0,1000 | 0,1414 | 0,0000 | 0,2000 |


### Tabel 4.19 Hasil evaluasi RAGAS mode LLM tanpa RAG

| Metrik | n | Rata-rata | Simpangan baku | Minimum | Maksimum |
|---|---|---|---|---|---|
| Answer relevancy | 0 | - | - | - | - |
| Factual correctness | 1 | 0,0000 | - | 0,0000 | 0,0000 |


### Tabel 4.20 Perbandingan skor dan hasil uji Wilcoxon

| Metrik | n pasangan | Dengan RAG | Tanpa RAG | Selisih | Statistik T | Nilai p | Keputusan | RAG lebih tinggi/lebih rendah/sama |
|---|---|---|---|---|---|---|---|---|
| Answer relevancy | 0 | - | - | - | - | - | tidak dapat diuji | 0/0/0 |
| Factual correctness | 1 | 0,0000 | 0,0000 | 0,0000 | - | - | tidak dapat diuji | 0/0/1 |


### Tabel 4.22 Waktu respons kedua mode (detik)

| Mode | Tahap | Rata-rata | Median | Persentil ke-95 | Maksimum |
|---|---|---|---|---|---|
| LLM dengan RAG | Retrieval | 1,931 | 1,931 | 2,766 | 2,859 |
| LLM dengan RAG | Pembangkitan | 2,257 | 2,257 | 2,361 | 2,372 |
| LLM dengan RAG | Total | 4,189 | 4,189 | 5,128 | 5,232 |
| LLM tanpa RAG | Pembangkitan | 4,184 | 4,184 | 5,548 | 5,700 |
| LLM tanpa RAG | Total | 4,185 | 4,185 | 5,549 | 5,701 |


### Rata-rata pemakaian token per pertanyaan

| Mode | Token masukan | Token keluaran | Token penalaran |
|---|---|---|---|
| LLM dengan RAG | 1802 | 583 | 379 |
| LLM tanpa RAG | 90 | 1606 | 80 |


### Tabel 4.21 Rata-rata skor per jenis pertanyaan

| Jenis | n | CP | CR | F | AR (RAG) | FC (RAG) | AR (tanpa RAG) | FC (tanpa RAG) |
|---|---|---|---|---|---|---|---|---|
| daftar | 0 | - | - | - | - | - | - | - |
| gejala | 1 | 1,0000 | - | 0,7500 | - | 0,0000 | - | 0,0000 |
| pengendalian | 1 | - | 1,0000 | - | - | 0,2000 | - | - |
| penyebab | 0 | - | - | - | - | - | - | - |


### Skor kosong (perlu dijalankan ulang atau dicatat sebagai keterbatasan)

- [rag] Apa gejala awal penyakit blas pada daun padi?: Context recall, Answer relevancy
- [rag] Bagaimana cara mengendalikan wereng batang cokelat?: Context precision, Faithfulness, Answer relevancy
- [llm] Apa gejala awal penyakit blas pada daun padi?: Answer relevancy
- [llm] Bagaimana cara mengendalikan wereng batang cokelat?: Answer relevancy, Factual correctness
