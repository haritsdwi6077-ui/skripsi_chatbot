"""Ingestion: baca PDF/TXT di data_source/, chunking, embedding, unggah ke Qdrant Cloud.

Dijalankan SECARA LOKAL (bukan di Vercel):

    python scripts/ingest.py --dry-run      # hitung halaman & chunk saja, tanpa API
    python scripts/ingest.py --recreate     # bangun ulang collection
    python scripts/ingest.py                # lanjutkan / tambah dokumen baru

Yang membuat proses lama BUKAN chunking (hanya hitungan milidetik), melainkan
embedding: setiap chunk harus dikirim ke API Gemini. Karena itu skrip ini:

1. Mengirim embedding secara batch (banyak chunk per permintaan HTTP).
2. Menyimpan setiap embedding ke cache lokal (.cache/), sehingga bila proses
   terhenti (kuota habis, internet putus) atau collection dibuat ulang,
   chunk yang sudah di-embedding TIDAK dikirim ulang ke API.
3. Memakai ID chunk yang tetap (deterministik), sehingga menjalankan ulang
   tidak menggandakan data di Qdrant.
4. Membaca subfolder (misalnya data_source/Blas/...).
5. Melaporkan PDF yang tidak punya teks (hasil scan) agar bisa di-OCR atau dibuang.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from rag_core import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_source"
CACHE_DIR = ROOT / ".cache"
NAMESPACE = uuid.UUID("7d3f5f0e-6a51-4c3e-9a0b-2b7f3c1d9e11")


def bersihkan(teks: str) -> str:
    """Membuang spasi berlebih dan pemenggalan kata di akhir baris."""
    teks = teks.replace("\r", "\n")
    teks = re.sub(r"-\n(?=\w)", "", teks)
    teks = re.sub(r"[ \t]+", " ", teks)
    teks = re.sub(r"\n{3,}", "\n\n", teks)
    return teks.strip()


def baca_dokumen(data_dir: Path | None = None) -> tuple[list[dict], list[str]]:
    """Mengembalikan (halaman, pdf_tanpa_teks). Subfolder ikut dibaca."""
    from pypdf import PdfReader

    data_dir = data_dir or DATA_DIR

    halaman, tanpa_teks = [], []
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        sumber = path.relative_to(data_dir).as_posix()
        akhiran = path.suffix.lower()
        if akhiran == ".pdf":
            try:
                reader = PdfReader(str(path))
                ada_teks = False
                for nomor, page in enumerate(reader.pages, start=1):
                    teks = bersihkan(page.extract_text() or "")
                    if len(teks) > 50:
                        ada_teks = True
                        halaman.append({"source": sumber, "page": nomor, "text": teks})
                if not ada_teks:
                    tanpa_teks.append(sumber)
            except Exception as exc:  # noqa: BLE001
                tanpa_teks.append(f"{sumber} (gagal dibaca: {exc.__class__.__name__})")
        elif akhiran in {".txt", ".md"}:
            teks = bersihkan(path.read_text(encoding="utf-8", errors="ignore"))
            if teks:
                halaman.append({"source": sumber, "page": None, "text": teks})
    return halaman, tanpa_teks


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


def buat_chunks(halaman: list[dict]) -> list[dict]:
    chunks = []
    for h in halaman:
        for urutan, bagian in enumerate(potong(h["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            kunci = f"{h['source']}|{h['page']}|{urutan}|{bagian}"
            chunks.append({
                "id": str(uuid.uuid5(NAMESPACE, kunci)),
                "hash": hashlib.sha1(bagian.encode("utf-8")).hexdigest(),
                "source": h["source"], "page": h["page"], "text": bagian,
            })
    return chunks


class CacheEmbedding:
    """Cache embedding per teks, disimpan sebagai JSON Lines (satu baris per chunk)."""

    def __init__(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        nama = f"embeddings_{config.EMBEDDING_MODEL}_{config.EMBEDDING_DIM}.jsonl"
        self.path = CACHE_DIR / nama
        self.data: dict[str, list[float]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for baris in f:
                    try:
                        rec = json.loads(baris)
                        self.data[rec["h"]] = rec["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # baris terakhir bisa terpotong bila proses dihentikan

    def simpan(self, hasil: dict[str, list[float]]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for h, v in hasil.items():
                f.write(json.dumps({"h": h, "v": [round(x, 7) for x in v]}) + "\n")
        self.data.update(hasil)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="hapus lalu buat ulang collection (cache embedding tetap dipakai)")
    parser.add_argument("--dry-run", action="store_true",
                        help="hanya baca PDF dan hitung chunk, tanpa memanggil API")
    parser.add_argument("--batch", type=int, default=50,
                        help="jumlah chunk per permintaan embedding (default 50)")
    parser.add_argument("--jeda", type=float, default=1.0,
                        help="jeda detik antar-permintaan embedding (atur sesuai kuota)")
    args = parser.parse_args()

    t0 = time.time()
    halaman, tanpa_teks = baca_dokumen()
    if not halaman:
        raise SystemExit(f"Tidak ada dokumen berteks di {DATA_DIR}")
    chunks = buat_chunks(halaman)
    jumlah_dok = len({c["source"] for c in chunks})
    print(f"{jumlah_dok} dokumen, {len(halaman)} halaman -> {len(chunks)} chunk "
          f"(baca + chunking {time.time() - t0:.1f} detik)")
    if tanpa_teks:
        print(f"\nPERINGATAN: {len(tanpa_teks)} PDF tidak menghasilkan teks "
              f"(kemungkinan hasil scan, perlu OCR atau dibuang):")
        for s in tanpa_teks:
            print("   -", s)

    cache = CacheEmbedding()
    belum = [c for c in chunks if c["hash"] not in cache.data]
    unik_belum = list({c["hash"]: c for c in belum}.values())
    n_req = -(-len(unik_belum) // args.batch) if unik_belum else 0
    print(f"\nSudah ada di cache: {len(chunks) - len(belum)} chunk | "
          f"perlu embedding: {len(unik_belum)} chunk unik "
          f"(~{n_req} permintaan batch @ {args.batch})")

    if args.dry_run:
        print("\nDry run selesai. Tidak ada API yang dipanggil.")
        return

    from qdrant_client import models  # noqa: PLC0415

    from database import vector_db  # noqa: PLC0415
    from rag_core import embeddings  # noqa: PLC0415

    # 1) Embedding yang belum ada di cache, per batch, dengan cache tersimpan tiap batch
    for i in range(0, len(unik_belum), args.batch):
        kelompok = unik_belum[i:i + args.batch]
        vektor = embeddings.embed_documents([c["text"] for c in kelompok])
        cache.simpan({c["hash"]: v for c, v in zip(kelompok, vektor)})
        selesai = min(i + args.batch, len(unik_belum))
        print(f"  embedding {selesai}/{len(unik_belum)}")
        if selesai < len(unik_belum) and args.jeda:
            time.sleep(args.jeda)

    # 2) Unggah ke Qdrant (ID tetap -> menjalankan ulang tidak menggandakan data)
    vector_db.ensure_collection(recreate=args.recreate)
    for i in range(0, len(chunks), 128):
        kelompok = chunks[i:i + 128]
        vector_db.upsert([
            models.PointStruct(
                id=c["id"],
                vector=cache.data[c["hash"]],
                payload={"source": c["source"], "page": c["page"], "text": c["text"]},
            )
            for c in kelompok
        ])
        print(f"  unggah {min(i + 128, len(chunks))}/{len(chunks)}")

    print(f"\nSelesai dalam {(time.time() - t0) / 60:.1f} menit. "
          f"Jumlah point dalam collection: {vector_db.count()}")


if __name__ == "__main__":
    main()
