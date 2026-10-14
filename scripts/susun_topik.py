"""Menyusun file di data_source/ ke subfolder topik, mengikuti folder rujukan.

Folder rujukan adalah folder yang sudah kamu kelompokkan per topik, misalnya:

    D:\\Koleksi_PDF\\Blas\\007_....pdf
    D:\\Koleksi_PDF\\HDB\\150_....pdf
    D:\\Koleksi_PDF\\Tungro\\044_....pdf

Skrip mencari file dengan nama yang sama di data_source/ lalu memindahkannya
ke data_source/<Topik>/. Isi folder rujukan tidak diubah.

    python scripts/susun_topik.py "D:\\Koleksi_PDF"              # pratinjau saja
    python scripts/susun_topik.py "D:\\Koleksi_PDF" --jalankan   # benar-benar pindahkan

Setelah itu jalankan:  python scripts/ingest.py --recreate
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_source"
EKSTENSI = {".pdf", ".txt", ".md"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rujukan", help="folder yang isinya sudah dikelompokkan per topik")
    parser.add_argument("--jalankan", action="store_true",
                        help="benar-benar memindahkan file (tanpa ini hanya pratinjau)")
    args = parser.parse_args()

    rujukan = Path(args.rujukan)
    if not rujukan.is_dir():
        raise SystemExit(f"Folder rujukan tidak ditemukan: {rujukan}")

    # nama file -> topik, dari folder rujukan (subfolder tingkat pertama = topik)
    topik_file: dict[str, str] = {}
    ganda: dict[str, set[str]] = defaultdict(set)
    for f in rujukan.rglob("*"):
        if f.is_file() and f.suffix.lower() in EKSTENSI:
            rel = f.relative_to(rujukan)
            if len(rel.parts) < 2:
                continue
            topik = rel.parts[0]
            if f.name in topik_file and topik_file[f.name] != topik:
                ganda[f.name].update({topik_file[f.name], topik})
            topik_file[f.name] = topik

    # file di data_source yang masih di luar subfolder
    lepas = sorted(f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in EKSTENSI)
    rencana, tanpa_topik = [], []
    for f in lepas:
        if f.name in topik_file:
            rencana.append((f, DATA_DIR / topik_file[f.name] / f.name))
        else:
            tanpa_topik.append(f.name)

    per_topik: dict[str, int] = defaultdict(int)
    for _, tujuan in rencana:
        per_topik[tujuan.parent.name] += 1

    print(f"File di data_source tanpa subfolder : {len(lepas)}")
    print(f"Ditemukan topiknya di folder rujukan: {len(rencana)}")
    for t in sorted(per_topik):
        print(f"   {t:20s} {per_topik[t]:3d} file")
    if ganda:
        print(f"\nPERHATIAN: {len(ganda)} file ada di lebih dari satu topik; "
              "dipakai topik yang terakhir ditemukan:")
        for nama, ts in sorted(ganda.items()):
            print(f"   - {nama}: {', '.join(sorted(ts))}")
    if tanpa_topik:
        print(f"\n{len(tanpa_topik)} file tidak ada di folder rujukan, tetap di data_source "
              "(pindahkan manual ke folder yang sesuai, misalnya Umum):")
        for nama in tanpa_topik:
            print(f"   - {nama}")

    if not args.jalankan:
        print("\nPratinjau saja, belum ada file yang dipindahkan. "
              "Tambahkan --jalankan untuk memindahkan.")
        return

    for asal, tujuan in rencana:
        tujuan.parent.mkdir(exist_ok=True)
        if tujuan.exists():
            print(f"   lewati, sudah ada: {tujuan.relative_to(DATA_DIR)}")
            continue
        shutil.move(str(asal), str(tujuan))
    print(f"\nSelesai memindahkan {len(rencana)} file. "
          "Lanjutkan dengan: python scripts/ingest.py --recreate")


if __name__ == "__main__":
    main()
