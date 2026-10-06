"""Koneksi ke vector database Qdrant Cloud.

Versi sebelumnya memakai QdrantClient(":memory:"), sehingga seluruh dokumen
harus di-embedding ulang setiap kali aplikasi dijalankan dan datanya hilang
saat proses berhenti. Versi ini memakai Qdrant Cloud, sehingga basis
pengetahuan cukup dibangun sekali melalui scripts/ingest.py dan dapat diakses
oleh aplikasi web yang di-deploy di Vercel.
"""

from functools import lru_cache

from qdrant_client import QdrantClient, models

from rag_core import config


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if not config.QDRANT_URL:
        raise RuntimeError("QDRANT_URL belum diatur (lihat .env.example).")
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY or None,
        timeout=config.REQUEST_TIMEOUT,
    )


def ensure_collection(recreate: bool = False) -> None:
    """Membuat collection bila belum ada; dipakai oleh skrip ingestion."""
    client = get_client()
    exists = client.collection_exists(config.QDRANT_COLLECTION)

    if recreate and exists:
        client.delete_collection(config.QDRANT_COLLECTION)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=config.EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
        )
        print(f"Collection '{config.QDRANT_COLLECTION}' dibuat "
              f"(dimensi {config.EMBEDDING_DIM}, jarak cosine).")


def upsert(points: list[models.PointStruct]) -> None:
    get_client().upsert(collection_name=config.QDRANT_COLLECTION, points=points)


def count() -> int:
    return get_client().count(config.QDRANT_COLLECTION, exact=True).count


def search(vector: list[float], top_k: int) -> list[dict]:
    """Mencari top-k potongan dokumen paling mirip dengan vektor kueri."""
    hits = get_client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", "-"),
            "page": hit.payload.get("page"),
            "score": hit.score,
        }
        for hit in hits
    ]

