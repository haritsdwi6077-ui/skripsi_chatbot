"""Membuat berkas jawaban baku: rag_core/data/jawaban_baku.json.

Setiap kueri baku (3 pertanyaan daftar + setiap hama/penyakit x aspek, 80 kueri)
dijawab SEKALI, lalu disimpan. Aplikasi web memakai jawaban dari berkas ini,
sehingga pertanyaan yang maknanya sama, misalnya "apa penyakit padi" dan
"penyakit padi apa aja", selalu mendapat jawaban dan rujukan yang persis sama
di server mana pun dan setelah server dinyalakan ulang.

Jalankan dari folder utama proyek (butuh .env berisi kunci Groq, Jina, Qdrant):

    python scripts/buat_jawaban_baku.py              # semua kueri, lanjutkan yang belum ada
    python scripts/buat_jawaban_baku.py --hanya-daftar
    python scripts/buat_jawaban_baku.py --ulang      # buat ulang semuanya

Setelah selesai, push berkas rag_core/data/jawaban_baku.json. Buat ulang berkas
ini bila dokumen rujukan di Qdrant diubah.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from rag_core import config, engine, simpanan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hanya-daftar", action="store_true",
                        help="hanya 3 pertanyaan daftar (penyakit, hama, keduanya)")
    parser.add_argument("--ulang", action="store_true", help="buat ulang jawaban yang sudah ada")
    parser.add_argument("--jeda", type=float, default=3.0,
                        help="jeda antarpermintaan (detik), agar tidak kena batas Groq")
    args = parser.parse_args()

    config.SIMPAN_JAWABAN = False   # jangan pakai simpanan lama saat membuat jawaban baru
    berkas = simpanan.BERKAS_JAWABAN
    data = {"jawaban": {}}
    if berkas.exists():
        data = json.loads(berkas.read_text(encoding="utf-8"))
    jawaban = data.setdefault("jawaban", {})

    daftar = list(engine.KUERI_DAFTAR.values()) if args.hanya_daftar else engine.semua_kueri_baku()
    for i, kueri in enumerate(daftar, start=1):
        if kueri in jawaban and not args.ulang:
            print(f"[{i}/{len(daftar)}] sudah ada: {kueri}")
            continue
        print(f"[{i}/{len(daftar)}] {kueri}")
        for percobaan in range(3):
            try:
                hasil = engine.answer(kueri, percakapan=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"   gagal ({e}); coba lagi dalam 20 detik")
                time.sleep(20)
        else:
            print("   dilewati")
            continue
        if hasil.get("pertanyaan_mandiri") != kueri or not hasil.get("contexts"):
            print("   dilewati: tidak ada dokumen yang cocok")
            continue
        jawaban[kueri] = {
            "answer": hasil["answer"],
            "contexts": hasil["contexts"],
            "usage": hasil.get("usage", {}),
        }
        data.update({"model": config.LLM_MODEL, "dibuat": datetime.now().isoformat(timespec="seconds")})
        berkas.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(args.jeda)

    print(f"\nSelesai: {len(jawaban)} jawaban di {berkas}")


if __name__ == "__main__":
    main()
