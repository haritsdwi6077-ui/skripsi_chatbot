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

# --- Embedding: memakai Gemini (penyedia LLM tidak wajib punya endpoint embedding) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

# --- Vector database: Qdrant Cloud ---
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "skripsi_knowledge_base")

# --- Parameter retrieval ---
TOP_K = int(os.environ.get("TOP_K", "4"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "180"))

REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "60"))

