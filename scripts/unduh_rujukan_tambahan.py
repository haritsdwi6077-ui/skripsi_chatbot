"""Unduh dokumen rujukan pengendalian tambahan (masukan validator Faperta).

Daftar dokumen ada di rag_core/data/rujukan_tambahan.csv. Jalankan di laptop:

    python scripts/unduh_rujukan_tambahan.py
    python scripts/ingest.py                 # tambahkan dokumen baru ke Qdrant
    python scripts/buat_jawaban_baku.py      # buat ulang jawaban baku

Berkas yang sudah ada dilewati. Bila sebuah tautan gagal diunduh, buka tautannya
di browser, simpan PDF-nya dengan nama dan folder yang tertera, lalu jalankan
ingest.
"""

import csv
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DAFTAR = ROOT / "rag_core" / "data" / "rujukan_tambahan.csv"


def main() -> None:
    gagal = []
    with DAFTAR.open(encoding="utf-8") as f:
        baris = list(csv.DictReader(f))
    for r in baris:
        tujuan = ROOT / "data_source" / r["berkas"]
        if tujuan.exists():
            print(f"sudah ada : {r['berkas']}")
            continue
        galat = None
        for _ in range(3):  # server kadang lambat: coba sampai tiga kali
            try:
                resp = requests.get(r["url"], timeout=180, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    raise ValueError("bukan berkas PDF")
                galat = None
                break
            except (requests.RequestException, ValueError) as e:
                galat = e
        if galat is not None:
            print(f"GAGAL     : {r['berkas']} ({galat})")
            gagal.append(r)
            continue
        tujuan.parent.mkdir(parents=True, exist_ok=True)
        tujuan.write_bytes(resp.content)
        print(f"diunduh   : {r['berkas']} ({len(resp.content) / 1e6:.1f} MB)")
    if gagal:
        print("\nUnduh manual lewat browser, lalu simpan dengan nama berikut:")
        for r in gagal:
            print(f"  data_source/{r['berkas']}\n    {r['url']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
