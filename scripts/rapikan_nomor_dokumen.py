"""Rapikan nomor urut dokumen sumber menjadi 001, 002, ... tanpa nomor yang loncat.

Daftar nama lama dan nama baru ada di data_source/peta_nama_dokumen.csv.
Urutannya mengikuti topik (Blas, HDB, Bercak_coklat, ..., Umum, Lain lain), lalu
nomor lama di dalam setiap topik. Empat dokumen di folder "Lain lain" yang
belum bernomor juga diberi nama yang jelas (penulis dan judul).

Jalankan sekali dari folder utama proyek:

    python scripts/rapikan_nomor_dokumen.py
    python scripts/ingest.py --recreate      # nama sumber di Qdrant ikut berganti
    python scripts/buat_jawaban_baku.py --ulang

Embedding tidak dikirim ulang ke Jina karena tersimpan di cache menurut isi
teksnya, jadi ingest ulang hanya membaca PDF dan mengunggah ke Qdrant.
Berkas peta_nama_dokumen.csv tetap disimpan sebagai jejak: hasil evaluasi lama
di folder hasil/ masih memakai nama lama.
"""

import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_source"
PETA = DATA / "peta_nama_dokumen.csv"
# Dikeluarkan dari basis pengetahuan: duplikat buku Suprihanto dkk. (sudah ada di
# folder Lain lain) dan dokumen hasil pindaian tanpa teks.
DIKELUARKAN = [
    DATA / "Umum" / "198_Pengendalian_Hama_dan_Penyakit_Utama_Tanaman_Padi_BSIP_Padi_2024.pdf",
    DATA / "WBC" / "061_Kerentanan_wereng_batang_cokelat_dari_enam_lokasi_di_Pulau_Jawa_terhadap_tiga_jenis_insekt.pdf",
]


def main() -> None:
    with PETA.open(encoding="utf-8") as f:
        peta = [(r["berkas_lama"], r["berkas_baru"]) for r in csv.DictReader(f)]

    for berkas in DIKELUARKAN:
        if berkas.exists():
            berkas.unlink()
            print(f"Dihapus: {berkas.relative_to(DATA)}")

    def tanpa_nomor(nama: str) -> str:
        return re.sub(r"^\d+[._ ]\s*", "", nama)

    ubah, sudah, hilang = [], 0, []
    for lama, baru in peta:
        if (DATA / baru).exists():
            sudah += 1
        elif (DATA / lama).exists():
            ubah.append((lama, baru))
        else:
            # Cadangan: berkas yang sama dengan nomor lain (misalnya dari versi
            # skrip sebelumnya) dikenali dari judulnya tanpa nomor urut
            folder, nama_baru = baru.split("/", 1)
            calon = [p.name for p in (DATA / folder).glob("*.pdf")
                     if tanpa_nomor(p.name) == tanpa_nomor(nama_baru)]
            if calon:
                ubah.append((f"{folder}/{calon[0]}", baru))
            else:
                hilang.append(lama)

    # Dua tahap (lewat nama sementara) agar tidak ada nama yang bertabrakan
    for i, (lama, _) in enumerate(ubah):
        os.rename(DATA / lama, DATA / f"{lama}.sementara{i}")
    for i, (lama, baru) in enumerate(ubah):
        os.rename(DATA / f"{lama}.sementara{i}", DATA / baru)

    print(f"Diganti namanya : {len(ubah)}")
    print(f"Sudah bernama baru: {sudah}")
    if hilang:
        print(f"Belum ada di folder ({len(hilang)}), misalnya dokumen tambahan yang belum diunduh:")
        for h in hilang:
            print(f"  {h}")
    for berkas in DIKELUARKAN:
        if berkas.exists():
            berkas.unlink()
            print(f"Dihapus: {berkas.relative_to(DATA)}")


if __name__ == "__main__":
    main()
