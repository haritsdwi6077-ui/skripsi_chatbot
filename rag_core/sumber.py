"""Tautan dan judul sumber rujukan yang ditampilkan di bawah jawaban.

Revisi seminar hasil: sebelumnya sumber hanya berupa nama berkas, sehingga
pengguna tidak bisa membuka dokumen aslinya. Sekarang setiap sumber diberi
tautan yang bisa diklik.

Urutan penentuan tautan untuk satu berkas:
  1. Kolom "tautan" di rag_core/data/tautan_sumber.csv, bila diisi. Isi dengan
     halaman resmi dokumen, misalnya DOI, situs jurnal, IRRI, atau FAO.
  2. Bila kosong, salinan PDF di repositori GitHub penelitian, dibuka lewat CDN
     jsDelivr (SUMBER_BASE_URL) langsung pada halaman yang dikutip (#page=N).

Judul: kolom "judul" di CSV bila diisi, selain itu nama berkas yang dirapikan
("138_Pengendalian_penyakit_blas.pdf" -> "Pengendalian penyakit blas").
CSV berada di rag_core/ karena folder data_source/ tidak ikut di-deploy ke Vercel.
"""

import csv
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from rag_core import config

BERKAS_CSV = Path(__file__).resolve().parent / "data" / "tautan_sumber.csv"


@lru_cache(maxsize=1)
def _daftar() -> dict[str, dict]:
    if not BERKAS_CSV.exists():
        return {}
    with BERKAS_CSV.open(encoding="utf-8") as f:
        return {
            (r.get("berkas") or "").strip(): {
                "judul": (r.get("judul") or "").strip(),
                "tautan": (r.get("tautan") or "").strip(),
            }
            for r in csv.DictReader(f)
            if (r.get("berkas") or "").strip()
        }


def judul_rapi(berkas: str) -> str:
    """Nama berkas tanpa folder, nomor urut, garis bawah, dan ekstensi."""
    nama = (berkas or "").rsplit("/", 1)[-1]
    nama = re.sub(r"\.pdf$", "", nama, flags=re.IGNORECASE)
    nama = re.sub(r"^\d+[_\-\s]+", "", nama)
    nama = re.sub(r"[_]+", " ", nama).strip()
    return nama or berkas


def tautan(berkas: str, halaman: int | None = None) -> str | None:
    """Tautan yang bisa diklik untuk satu sumber, atau None bila tidak tersedia."""
    if not berkas or berkas == "-":
        return None
    resmi = _daftar().get(berkas, {}).get("tautan")
    if resmi:
        return resmi
    dasar = (config.SUMBER_BASE_URL or "").strip()
    if not dasar:
        return None
    url = dasar.rstrip("/") + "/" + quote(berkas, safe="/")
    return f"{url}#page={halaman}" if halaman else url


def judul(berkas: str) -> str:
    return _daftar().get(berkas, {}).get("judul") or judul_rapi(berkas)


def rincian(nomor: int, ctx: dict) -> dict:
    """Satu sumber untuk tampilan: nomor rujukan, judul, halaman, tautan, nama berkas."""
    berkas = ctx.get("source") or ""
    halaman = ctx.get("page")
    return {
        "nomor": nomor,
        "judul": judul(berkas),
        "halaman": halaman,
        "tautan": tautan(berkas, halaman),
        "berkas": berkas,
    }
