"""Melihat potongan dokumen yang diambil untuk sebuah pertanyaan.

Berguna untuk memeriksa kenapa sebuah jawaban hanya membahas satu topik, dan
untuk lampiran skripsi (contoh hasil retrieval).

    python scripts/cek_retrieval.py "apa saja penyakit utama pada padi?"
    python scripts/cek_retrieval.py "gejala penyakit blas" --kandidat 40

Kolom: urutan, skor kemiripan (cosine), topik (subfolder), dokumen, halaman.
Tanda >> menandai potongan yang akhirnya dipilih dan dikirim ke LLM.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from database import vector_db  # noqa: E402
from rag_core import config, engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pertanyaan")
    parser.add_argument("--kandidat", type=int, default=config.RETRIEVAL_KANDIDAT)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config.RETRIEVAL_KANDIDAT = args.kandidat
    k, kandidat, dipilih = engine.ambil_konteks(args.pertanyaan, args.top_k)
    kunci = {(d["source"], d["page"], d["text"][:80]) for d in dipilih}

    print(f"\nPertanyaan: {args.pertanyaan}")
    per_topik = k > config.TOP_K and bool(vector_db.daftar_topik())
    cara = "satu potongan per topik" if per_topik else f"MMR, jendela skor {config.JENDELA_SKOR}"
    print(f"{len(kandidat)} kandidat, dipilih {k} ({cara}):\n")
    for i, k in enumerate(kandidat, 1):
        tanda = ">>" if (k["source"], k["page"], k["text"][:80]) in kunci else "  "
        topik = engine.topik_dari(k["source"]) or "-"
        print(f"{tanda} {i:2d}. {k['score']:.3f}  {topik[:18]:18s}  "
              f"{Path(k['source']).name[:60]}  hlm. {k['page']}")

    sebaran = Counter(engine.topik_dari(k["source"]) or "-" for k in kandidat)
    print("\nSebaran topik di kandidat:", dict(sebaran.most_common()))
    print("Topik yang dipilih       :", [d["topik"] or "-" for d in dipilih])
    if all(engine.topik_dari(k["source"]) is None for k in kandidat):
        print("\nCATATAN: sumber belum punya subfolder topik. Susun PDF di "
              "data_source/<Topik>/ lalu jalankan ingest dengan --recreate.")


if __name__ == "__main__":
    main()
