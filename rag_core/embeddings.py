"""Pembuatan embedding teks.

DeepSeek hanya menyediakan endpoint chat completions dan tidak memiliki
endpoint embedding, sehingga tahap embedding pada pipeline RAG ini memakai
model embedding Gemini yang diakses melalui REST API.

Model dan dimensi embedding HARUS sama antara tahap ingestion dan tahap query.
"""

"""Pembuatan embedding teks memakai model embedding Gemini."""

import math

import requests

from rag_core import config

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


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
    response = requests.post(
        url,
        headers={"x-goog-api-key": config.GEMINI_API_KEY},
        json=payload,
        timeout=config.REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(
            f"Gemini embedContent {response.status_code} "
            f"(model={config.EMBEDDING_MODEL}, dim={config.EMBEDDING_DIM}): "
            f"{response.text[:500]}"
        )

    vector = response.json()["embedding"]["values"]
    if config.EMBEDDING_DIM != 3072:
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vector = [v / norm for v in vector]
    return vector


def embed_document(text: str) -> list[float]:
    """Embedding untuk potongan dokumen yang disimpan di vector database."""
    return _embed(text, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embedding untuk pertanyaan pengguna."""
    return _embed(text, "RETRIEVAL_QUERY")

