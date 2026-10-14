"""Konfigurasi terpusat. Semua kredensial dibaca dari environment variable."""

import os


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Environment variable {name} belum diatur.")
    return value


# --- LLM (generation): Groq, endpoint kompatibel OpenAI ---
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
# Cek nama model yang masih aktif di https://console.groq.com/docs/models sebelum deploy.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
# Khusus model reasoning (gpt-oss di Groq): "low" | "medium" | "high".
# Kosongkan untuk model non-reasoning; parameter ini lalu tidak dikirim.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "").strip()
# Batas token jawaban (0 = tidak dibatasi / pakai bawaan penyedia).
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "0"))

# --- Embedding: memakai Jina AI (penyedia LLM tidak wajib punya endpoint embedding) ---
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "https://api.jina.ai/v1/embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
# jina-embeddings-v5: 1-1024 (Matryoshka). Ubah nilai ini = wajib ingest ulang --recreate.
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

# --- Vector database: Qdrant Cloud ---
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "skripsi_knowledge_base")

# --- Parameter retrieval ---
TOP_K = int(os.environ.get("TOP_K", "4"))
# Pertanyaan bertipe daftar ("apa saja", "sebutkan", "jenis-jenis") butuh lebih
# banyak potongan agar beberapa hama/penyakit bisa muncul. 0 = sama dengan TOP_K.
TOP_K_DAFTAR = int(os.environ.get("TOP_K_DAFTAR", "8"))
# Retrieval beragam (MMR): ambil RETRIEVAL_KANDIDAT potongan termirip dari Qdrant,
# lalu pilih TOP_K yang relevan TETAPI tidak saling mirip, maksimal MAKS_PER_SUMBER
# potongan dari dokumen yang sama. MMR_LAMBDA: 1 = murni relevansi, 0 = murni keragaman.
MMR_AKTIF = os.environ.get("MMR_AKTIF", "1").strip() != "0"
RETRIEVAL_KANDIDAT = int(os.environ.get("RETRIEVAL_KANDIDAT", "40"))
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.6"))
MAKS_PER_SUMBER = int(os.environ.get("MAKS_PER_SUMBER", "2"))
# Keragaman topik: topik = nama subfolder di data_source (Blas, HDB, Tungro, ...).
# Kandidat dengan skor dalam JENDELA_SKOR dari skor tertinggi dianggap sama
# relevannya; di antara mereka, tiap topik lebih dulu diberi 1 tempat. Pertanyaan
# spesifik tidak terpengaruh, karena topik lain biasanya jauh di luar jendela.
JENDELA_SKOR = float(os.environ.get("JENDELA_SKOR", "0.05"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "180"))

REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "60"))

