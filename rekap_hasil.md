# Rekap hasil evaluasi

Berkas: hasil\glm_rag_v2.csv, hasil\glm_llm.csv

Jumlah pertanyaan: mode RAG = 40, mode tanpa RAG = 40

### Tabel 4.18 Hasil evaluasi RAGAS mode LLM dengan RAG

| Metrik | n | Rata-rata | Median | Simpangan baku | Minimum | Maksimum | Jumlah skor 0 |
|---|---|---|---|---|---|---|---|
| Context precision | 40 | 0,7209 | 0,8194 | 0,3134 | 0,0000 | 1,0000 | 3 |
| Context recall | 40 | 0,7729 | 1,0000 | 0,4039 | 0,0000 | 1,0000 | 8 |
| Faithfulness | 40 | 0,8678 | 1,0000 | 0,2197 | 0,1250 | 1,0000 | 0 |
| Answer relevancy | 40 | 0,8363 | 0,8423 | 0,0679 | 0,6688 | 0,9526 | 0 |
| Factual correctness | 40 | 0,4650 | 0,5000 | 0,2651 | 0,0000 | 0,9200 | 4 |


### Tabel 4.19 Hasil evaluasi RAGAS mode LLM tanpa RAG

| Metrik | n | Rata-rata | Median | Simpangan baku | Minimum | Maksimum | Jumlah skor 0 |
|---|---|---|---|---|---|---|---|
| Answer relevancy | 40 | 0,7971 | 0,8332 | 0,2032 | 0,0000 | 1,0000 | 2 |
| Factual correctness | 40 | 0,2290 | 0,0650 | 0,3255 | 0,0000 | 0,9600 | 17 |


### Tabel 4.20 Perbandingan skor dan hasil uji Wilcoxon

| Metrik | n pasangan | Dengan RAG | Tanpa RAG | Selisih | Statistik T | Nilai p | Ukuran efek r | Kategori efek | Keputusan | RAG lebih tinggi/lebih rendah/sama |
|---|---|---|---|---|---|---|---|---|---|---|
| Answer relevancy | 40 | 0,8363 | 0,7971 | 0,0392 | 334,0 | 0,3138 | 0,159 | kecil | H0 diterima | 22/18/0 |
| Factual correctness | 40 | 0,4650 | 0,2290 | 0,2360 | 97,0 | 0,0001 | 0,643 | besar | H0 ditolak | 31/7/2 |


### Uji sensitivitas: 12 soal dikeluarkan (no. 1, 5, 6, 7, 11, 13, 21, 28, 31, 34, 37, 40), 28 soal tersisa

| Metrik | n pasangan | Dengan RAG | Tanpa RAG | Selisih | Statistik T | Nilai p | Ukuran efek r | Kategori efek | Keputusan | RAG lebih tinggi/lebih rendah/sama |
|---|---|---|---|---|---|---|---|---|---|---|
| Answer relevancy | 28 | 0,8323 | 0,7753 | 0,0570 | 150,0 | 0,2359 | 0,224 | kecil | H0 diterima | 17/11/0 |
| Factual correctness | 28 | 0,5054 | 0,3250 | 0,1804 | 75,0 | 0,0036 | 0,551 | besar | H0 ditolak | 22/6/0 |


### Tabel 4.22 Waktu respons kedua mode (detik)

| Mode | Tahap | Rata-rata | Median | Persentil ke-95 | Maksimum |
|---|---|---|---|---|---|
| LLM dengan RAG | Retrieval | 4,174 | 2,917 | 8,753 | 16,826 |
| LLM dengan RAG | Pembangkitan | 2,152 | 1,953 | 3,635 | 5,170 |
| LLM dengan RAG | Total | 6,328 | 5,207 | 10,987 | 18,505 |
| LLM tanpa RAG | Pembangkitan | 3,506 | 3,265 | 6,245 | 7,033 |
| LLM tanpa RAG | Total | 3,507 | 3,266 | 6,246 | 7,034 |


### Rata-rata pemakaian token per pertanyaan

| Mode | Token masukan | Token keluaran | Token penalaran |
|---|---|---|---|
| LLM dengan RAG | 2666 | 432 | 257 |
| LLM tanpa RAG | 92 | 1355 | 162 |


### Tabel 4.21 Rata-rata skor per jenis pertanyaan

| Jenis | n | CP | CR | F | AR (RAG) | FC (RAG) | AR (tanpa RAG) | FC (tanpa RAG) |
|---|---|---|---|---|---|---|---|---|
| daftar | 6 | 0,5681 | 0,6667 | 0,6389 | 0,8939 | 0,5350 | 0,8867 | 0,3433 |
| gejala | 12 | 0,8628 | 0,8333 | 0,9907 | 0,8291 | 0,5100 | 0,8476 | 0,3033 |
| pengendalian | 12 | 0,5495 | 0,6181 | 0,8324 | 0,7994 | 0,3842 | 0,6517 | 0,1125 |
| penyebab | 10 | 0,8481 | 0,9500 | 0,9000 | 0,8548 | 0,4660 | 0,8573 | 0,2110 |

