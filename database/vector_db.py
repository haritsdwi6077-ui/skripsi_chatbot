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

    # Indeks kata kunci untuk field "topik", dipakai untuk menyaring pencarian per
    # topik dan untuk mendaftar topik yang ada. Aman dipanggil berulang kali.
    try:
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name="topik",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Peringatan: indeks topik tidak dibuat ({exc.__class__.__name__}).")


def upsert(points: list[models.PointStruct]) -> None:
    get_client().upsert(collection_name=config.QDRANT_COLLECTION, points=points)


def count() -> int:
    return get_client().count(config.QDRANT_COLLECTION, exact=True).count


@lru_cache(maxsize=1)
def daftar_topik() -> tuple[str, ...]:
    """Topik (subfolder data_source) yang ada di collection, tanpa "-".

    Memakai facet Qdrant pada field "topik". Kosong bila collection dibangun
    tanpa subfolder atau server tidak mendukung facet.
    """
    try:
        hasil = get_client().facet(
            collection_name=config.QDRANT_COLLECTION, key="topik", limit=200
        )
        return tuple(sorted(str(h.value) for h in hasil.hits if str(h.value) != "-"))
    except Exception:  # noqa: BLE001
        return ()


def search(vector: list[float], top_k: int, with_vectors: bool = False,
           topik: str | None = None) -> list[dict]:
    """Mencari top-k potongan dokumen paling mirip dengan vektor kueri.

    with_vectors=True ikut mengembalikan vektor tiap potongan (kunci "vector"),
    dipakai untuk memilih potongan yang beragam (MMR) di engine.
    topik: bila diisi, pencarian dibatasi pada satu topik (subfolder).
    """
    saring = None
    if topik:
        saring = models.Filter(must=[
            models.FieldCondition(key="topik", match=models.MatchValue(value=topik))
        ])
    hits = get_client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=with_vectors,
        query_filter=saring,
    ).points

    hasil = []
    for hit in hits:
        item = {
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", "-"),
            "page": hit.payload.get("page"),
            "score": hit.score,
        }
        if with_vectors:
            item["vector"] = hit.vector
        hasil.append(item)
    return hasil

