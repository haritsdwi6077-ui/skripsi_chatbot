"""Pembuatan embedding teks.

DeepSeek hanya menyediakan endpoint chat completions dan tidak memiliki
endpoint embedding, sehingga tahap embedding pada pipeline RAG ini memakai
model embedding Gemini yang diakses melalui REST API.

Model dan dimensi embedding HARUS sama antara tahap ingestion dan tahap query.
"""

import math
import time

import requests

from rag_core import config

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _normalisasi(vector: list[float]) -> list[float]:
    """Vektor dengan dimensi < 3072 perlu dinormalisasi ulang (panjang = 1)."""
    if config.EMBEDDING_DIM != 3072:
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vector = [v / norm for v in vector]
    return vector


def _post_dengan_retry(url: str, payload: dict, percobaan: int = 6) -> dict:
    """POST ke Gemini; bila kena rate limit (429) atau galat server, tunggu lalu ulangi."""
    jeda = 5.0
    for ke in range(1, percobaan + 1):
        response = requests.post(
            url,
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json=payload,
            timeout=config.REQUEST_TIMEOUT,
        )
        if response.ok:
            return response.json()
        if response.status_code in _RETRY_STATUS and ke < percobaan:
            tunggu = float(response.headers.get("Retry-After") or jeda)
            print(f"    Gemini {response.status_code}, menunggu {tunggu:.0f} detik "
                  f"lalu mencoba lagi ({ke}/{percobaan - 1})...")
            time.sleep(tunggu)
            jeda = min(jeda * 2, 120)
            continue
        raise RuntimeError(
            f"Gemini {response.status_code} "
            f"(model={config.EMBEDDING_MODEL}, dim={config.EMBEDDING_DIM}): "
            f"{response.text[:500]}"
        )
    raise RuntimeError("Gemini: percobaan ulang habis.")


def _embed(text: str, task_type: str) -> list[float]:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY belum diatur.")

    url = f"{_BASE}/{config.EMBEDDING_MODEL}:embedContent"
    payload = {
        "model": f"models/{config.EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": config.EMBEDDING_DIM,
    }
    data = _post_dengan_retry(url, payload)
    return _normalisasi(data["embedding"]["values"])


def embed_document(text: str) -> list[float]:
    """Embedding untuk potongan dokumen yang disimpan di vector database."""
    return _embed(text, "RETRIEVAL_DOCUMENT")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embedding banyak potongan dokumen dalam SATU permintaan (batchEmbedContents).

    Jauh lebih cepat daripada memanggil embed_document() satu per satu karena
    jumlah permintaan HTTP berkurang drastis. Bila endpoint batch menolak
    (misalnya format berubah), otomatis kembali ke cara satu per satu.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY belum diatur.")
    if not texts:
        return []

    url = f"{_BASE}/{config.EMBEDDING_MODEL}:batchEmbedContents"
    payload = {
        "requests": [
            {
                "model": f"models/{config.EMBEDDING_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": config.EMBEDDING_DIM,
            }
            for t in texts
        ]
    }
    try:
        data = _post_dengan_retry(url, payload)
        vektor = [e["values"] for e in data["embeddings"]]
        if len(vektor) != len(texts):
            raise RuntimeError("jumlah embedding tidak sama dengan jumlah teks")
    except RuntimeError as exc:
        if " 429 " in f" {exc} ":
            raise  # kuota habis: jangan dibanjiri permintaan satu per satu
        print(f"    batchEmbedContents gagal ({str(exc)[:120]}); beralih ke satu per satu.")
        return [embed_document(t) for t in texts]
    return [_normalisasi(v) for v in vektor]


def embed_query(text: str) -> list[float]:
    """Embedding untuk pertanyaan pengguna."""
    return _embed(text, "RETRIEVAL_QUERY")

