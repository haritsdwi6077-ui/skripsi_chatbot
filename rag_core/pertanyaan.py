"""Melengkapi pertanyaan yang terpotong sebelum pencarian dokumen.

Masalah yang ditangani: pertanyaan pertama seperti "PADI ADALAH" atau
"pengertian blas" dicari apa adanya, sehingga potongan dokumen yang berisi
pengertiannya tidak terambil. Pertanyaan yang sama baru terjawab bila ditanyakan
lagi, karena penulisan ulang pertanyaan oleh LLM hanya berjalan bila sudah ada
riwayat percakapan.

Fungsi ini memakai aturan, tanpa LLM, dan hanya dipakai untuk pertanyaan tanpa
riwayat:
  1. Pertanyaan yang seluruhnya huruf kapital ditulis dengan huruf biasa.
  2. Pola pengertian diubah menjadi "Apa itu X?", misalnya "X adalah",
     "X itu", "pengertian X", atau "definisi X".
  3. Nama hama atau penyakit saja (paling banyak 4 kata, tanpa kata tanya),
     misalnya "wereng cokelat", diubah menjadi "Apa itu wereng cokelat?".
Pertanyaan yang sudah lengkap tidak diubah. Seluruh 40 pertanyaan test set juga
tidak berubah (lihat tests/test_pertanyaan.py), jadi hasil evaluasi RAGAS tetap
sama.
"""

import re

from rag_core.saran import cari_topik

# Kata yang menandakan pertanyaan sudah lengkap
KATA_TANYA = re.compile(
    r"\b(apa|apakah|bagaimana|gimana|kenapa|mengapa|kapan|dimana|di\s+mana|berapa|siapa|"
    r"mana|sebutkan|jelaskan|bisakah|bolehkah|cara|ciri|tanda|gejala|penyebab|"
    r"mengendalikan|pengendalian|dikendalikan|mencegah|pencegahan|mengatasi)\b",
    re.IGNORECASE,
)
POLA_AKHIRAN = re.compile(
    r"^(?P<x>.+?)\s+(adalah|itu|yaitu|ialah|artinya|maksudnya|merupakan)\s*[?.!]*$",
    re.IGNORECASE,
)
POLA_AWALAN = re.compile(
    r"^(pengertian|definisi|arti|maksud)\s+(dari\s+)?(?P<x>.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
# Singkatan yang tetap ditulis kapital setelah huruf kapital semua diubah
SINGKATAN = {"hdb": "HDB", "wbc": "WBC", "pht": "PHT", "popt": "POPT", "bph": "BPH", "blb": "BLB"}


def _huruf_biasa(teks: str) -> str:
    teks = teks.lower()
    teks = re.sub(r"\b[a-z]+\b", lambda m: SINGKATAN.get(m.group(0), m.group(0)), teks)
    return teks[:1].upper() + teks[1:]


def _apa_itu(x: str) -> str:
    x = x.strip().rstrip("?.!").strip()
    if x[:1].isupper() and not x[:2].isupper():
        x = x[:1].lower() + x[1:]
    return f"Apa itu {x}?"


def lengkapi_pertanyaan(pertanyaan: str) -> str:
    """Pertanyaan yang siap dipakai untuk pencarian dokumen."""
    teks = re.sub(r"\s+", " ", (pertanyaan or "").strip())
    huruf = [c for c in teks if c.isalpha()]
    if len(huruf) >= 3 and all(c.isupper() for c in huruf):
        teks = _huruf_biasa(teks)
    if not teks or KATA_TANYA.search(teks):
        return teks
    m = POLA_AWALAN.match(teks) or POLA_AKHIRAN.match(teks)
    if m and m.group("x").strip():
        return _apa_itu(m.group("x"))
    if len(teks.split()) <= 4 and cari_topik(teks):
        return _apa_itu(teks)
    return teks
