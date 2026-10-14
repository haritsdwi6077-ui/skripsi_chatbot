"""Pembuatan embedding teks dengan Jina AI Embeddings API.

Penyedia LLM (Groq/DeepSeek) tidak menyediakan endpoint embedding, sehingga
tahap embedding pada pipeline RAG ini memakai model embedding Jina yang
diakses melalui REST API (https://api.jina.ai/v1/embeddings).

Model jina-embeddings-v5 memakai adapter tugas yang berbeda untuk dokumen
("retrieval.passage") dan pertanyaan ("retrieval.query"). Model, dimensi, dan
pasangan task ini HARUS sama antara tahap ingestion dan tahap query.
"""

import time

import requests

from rag_core import config

_RETRY_STATUS = {429, 500, 502, 503, 504}

TASK_DOKUMEN = "retrieval.passage"
TASK_KUERI = "retrieval.query"


def _post_dengan_retry(payload: dict, percobaan: int = 6) -> dict:
    """POST ke Jina; bila kena rate limit (429) atau galat server, tunggu lalu ulangi."""
    jeda = 5.0
    for ke in range(1, percobaan + 1):
        try:
            response = requests.post(
                config.JINA_BASE_URL,
                headers={
                    "Authorization": f"Bearer {config.JINA_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            if ke < percobaan:
                print(f"    Koneksi ke Jina terputus, mencoba lagi dalam {jeda:.0f} detik "
                      f"({ke}/{percobaan - 1})...")
                time.sleep(jeda)
                jeda = min(jeda * 2, 120)
                continue
            raise RuntimeError(
                f"Tidak bisa terhubung ke Jina ({config.JINA_BASE_URL}). "
                f"Periksa koneksi internet. Detail: {exc}"
            ) from exc
        if response.ok:
            return response.json()
        if response.status_code in _RETRY_STATUS and ke < percobaan:
            tunggu = float(response.headers.get("Retry-After") or jeda)
            print(f"    Jina {response.status_code}, menunggu {tunggu:.0f} detik "
                  f"lalu mencoba lagi ({ke}/{percobaan - 1})...")
            time.sleep(tunggu)
            jeda = min(jeda * 2, 120)
            continue
        raise RuntimeError(
            f"Jina {response.status_code} "
            f"(model={config.EMBEDDING_MODEL}, dim={config.EMBEDDING_DIM}): "
            f"{response.text[:500]}"
        )
    raise RuntimeError("Jina: percobaan ulang habis.")


def embed_texts(texts: list[str], task: str, percobaan: int = 6) -> list[list[float]]:
    """Embedding banyak teks dalam SATU permintaan. Urutan hasil = urutan input.

    percobaan: jumlah maksimal percobaan bila kena rate limit atau koneksi putus.
    """
    if not config.JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY belum diatur.")
    if not texts:
        return []

    payload = {
        "model": config.EMBEDDING_MODEL,
        "task": task,
        "dimensions": config.EMBEDDING_DIM,
        "normalized": True,   # panjang vektor = 1, cocok untuk jarak cosine di Qdrant
        "truncate": True,     # teks terlalu panjang dipotong, bukan menyebabkan galat
        "embedding_type": "float",
        "input": texts,
    }
    data = _post_dengan_retry(payload, percobaan)
    hasil = sorted(data["data"], key=lambda d: d["index"])
    vektor = [d["embedding"] for d in hasil]
    if len(vektor) != len(texts):
        raise RuntimeError(
            f"Jina mengembalikan {len(vektor)} embedding untuk {len(texts)} teks."
        )
    return vektor


def embed_document(text: str) -> list[float]:
    """Embedding untuk satu potongan dokumen yang disimpan di vector database."""
    return embed_texts([text], TASK_DOKUMEN)[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embedding banyak potongan dokumen dalam satu permintaan HTTP."""
    return embed_texts(texts, TASK_DOKUMEN)


def embed_query(text: str) -> list[float]:
    """Embedding untuk pertanyaan pengguna.

    Hanya 3 percobaan (jeda total sekitar 15 detik) agar pengguna web tidak
    menunggu terlalu lama dan tetap di bawah batas 60 detik Vercel.
    """
    return embed_texts([text], TASK_KUERI, percobaan=3)[0]
