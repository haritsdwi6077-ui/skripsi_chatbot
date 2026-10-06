"""Ingestion: baca PDF/TXT di data_source/, chunking, embedding, unggah ke Qdrant Cloud.

Dijalankan SECARA LOKAL (bukan di Vercel):

    python scripts/ingest.py --recreate

Jalankan ulang hanya bila dokumen, model embedding, atau EMBEDDING_DIM berubah.
"""

import argparse
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from qdrant_client import models  # noqa: E402

from database import vector_db  # noqa: E402
from rag_core import config, embeddings  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data_source"


def bersihkan(teks: str) -> str:
    """Membuang spasi berlebih dan pemenggalan kata di akhir baris."""
    teks = teks.replace("\r", "\n")
    teks = re.sub(r"-\n(?=\w)", "", teks)
    teks = re.sub(r"[ \t]+", " ", teks)
    teks = re.sub(r"\n{3,}", "\n\n", teks)
    return teks.strip()


def baca_dokumen() -> list[dict]:
    """Mengembalikan daftar {'source', 'page', 'text'} dari isi data_source/."""
    from pypdf import PdfReader

    halaman = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for nomor, page in enumerate(reader.pages, start=1):
                teks = bersihkan(page.extract_text() or "")
                if len(teks) > 50:
                    halaman.append({"source": path.name, "page": nomor, "text": teks})
        elif path.suffix.lower() in {".txt", ".md"}:
            teks = bersihkan(path.read_text(encoding="utf-8"))
            if teks:
                halaman.append({"source": path.name, "page": None, "text": teks})
    return halaman


def potong(teks: str, ukuran: int, tumpang_tindih: int) -> list[str]:
    """Chunking berbasis karakter dengan overlap, dipotong pada batas spasi."""
    potongan, mulai = [], 0
    while mulai < len(teks):
        akhir = min(mulai + ukuran, len(teks))
        if akhir < len(teks):
            spasi = teks.rfind(" ", mulai + int(ukuran * 0.6), akhir)
            if spasi != -1:
                akhir = spasi
        bagian = teks[mulai:akhir].strip()
        if len(bagian) > 40:
            potongan.append(bagian)
        if akhir >= len(teks):
            break
        mulai = max(akhir - tumpang_tindih, mulai + 1)
    return potongan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="hapus lalu buat ulang collection")
    parser.add_argument("--jeda", type=float, default=0.5,
                        help="jeda detik antar-batch agar aman dari rate limit")
    args = parser.parse_args()

    halaman = baca_dokumen()
    if not halaman:
        raise SystemExit(f"Tidak ada dokumen di {DATA_DIR}")

    chunks = []
    for h in halaman:
        for bagian in potong(h["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunks.append({"source": h["source"], "page": h["page"], "text": bagian})
    print(f"{len(halaman)} halaman -> {len(chunks)} chunk")

    vector_db.ensure_collection(recreate=args.recreate)

    batch, terkirim = [], 0
    for i, chunk in enumerate(chunks, start=1):
        batch.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings.embed_document(chunk["text"]),
                payload=chunk,
            )
        )
        if len(batch) == 16 or i == len(chunks):
            vector_db.upsert(batch)
            terkirim += len(batch)
            batch = []
            print(f"  {terkirim}/{len(chunks)} chunk terunggah")
            if i < len(chunks) and args.jeda:
                time.sleep(args.jeda)

    print("Selesai. Jumlah point dalam collection:", vector_db.count())


if __name__ == "__main__":
    main()

