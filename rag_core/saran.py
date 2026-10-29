"""Saran pertanyaan lanjutan yang ditampilkan di bawah jawaban.

Menanggapi masukan responden kuesioner: pengguna merasa harus mengetik ulang
pertanyaan berkali-kali, dan meminta rekomendasi pertanyaan lanjutan.

Saran disusun dengan aturan, tanpa memanggil LLM, sehingga:
  - tidak menambah waktu tunggu atau pemakaian kuota API,
  - hanya menyarankan hama dan penyakit yang memang ada di basis pengetahuan,
  - hasilnya bisa diuji dan dijelaskan (lihat tests/test_saran.py).

Cara kerja:
  1. Cari nama hama atau penyakit di pertanyaan mandiri (hasil penulisan ulang).
     Bila tidak ada, misalnya pengguna hanya menceritakan gejala, cari di jawaban.
  2. Kenali aspek yang sedang ditanyakan: gejala, penyebab, pengendalian,
     pencegahan, atau musuh alami.
  3. Sarankan aspek lain untuk hama atau penyakit yang sama.
  Pertanyaan bertipe daftar ("penyakit padi apa saja?") diberi saran gejala untuk
  beberapa hama atau penyakit yang disebut di jawaban. Bila tidak ada yang
  dikenali, misalnya jawaban menolak pertanyaan di luar padi, diberikan contoh
  pertanyaan umum agar pengguna tahu apa yang bisa ditanyakan.
"""

import re

MAKS_SARAN = 3

# (nama yang ditampilkan, jenis, pola pengenal). Urutan tidak berpengaruh:
# yang dipakai adalah nama yang paling awal muncul di teks.
TOPIK = [
    ("penyakit blas", "penyakit", r"\bblas(?:t)?\b|busuk leher|pyricularia|magnaporthe"),
    ("penyakit hawar daun bakteri (kresek)", "penyakit",
     r"\bkresek\b|hawar daun bakteri|\bhdb\b|xanthomonas|bacterial leaf blight"),
    ("penyakit bercak cokelat", "penyakit",
     r"bercak (?:daun )?cokelat|brown spot|bipolaris|helminthosporium|cochliobolus"),
    ("penyakit hawar pelepah", "penyakit", r"hawar pelepah|busuk pelepah|rhizoctonia|sheath blight"),
    ("penyakit tungro", "penyakit", r"\btungro\b"),
    ("penyakit kerdil rumput", "penyakit", r"kerdil rumput|grassy stunt"),
    ("penyakit kerdil hampa", "penyakit", r"kerdil hampa|ragged stunt"),
    ("wereng batang cokelat", "hama",
     r"wereng (?:batang )?cokelat|\bwbc\b|nilaparvata|brown planthopper"),
    ("wereng hijau", "hama", r"wereng hijau|nephotettix|green leafhopper"),
    ("penggerek batang padi", "hama",
     r"penggerek batang|\bsundep\b|\bbeluk\b|scirpophaga|\bchilo\b|stem borer"),
    ("walang sangit", "hama", r"walang sangit|leptocorisa"),
    ("tikus sawah", "hama", r"\btikus\b"),
    ("keong mas", "hama", r"keong mas|pomacea|golden apple snail"),
    ("hama putih palsu", "hama", r"putih palsu|pelipat daun|cnaphalocrocis"),
]
_TOPIK = [(nama, jenis, re.compile(pola)) for nama, jenis, pola in TOPIK]

# Aspek pertanyaan. Urutan menentukan prioritas bila beberapa cocok,
# misalnya "cara mencegah serangan" dihitung sebagai pencegahan.
ASPEK = [
    ("musuh_alami", re.compile(r"musuh alami|predator|parasitoid")),
    ("pencegahan", re.compile(r"cegah|pencegahan|antisipasi")),
    ("pengendalian", re.compile(
        # "endali" menangkap kendali, mengendalikan, dikendalikan, dan pengendalian
        r"endali|atasi|basmi|berantas|obat|insektisida|"
        r"fungisida|bakterisida|pestisida|semprot")),
    ("penyebab", re.compile(r"penyebab|disebabkan|sebabnya|patogen|apa itu")),
    ("gejala", re.compile(r"gejala|tanda|ciri|seperti apa|bercak|serangan")),
]

TEMPLAT = {
    "penyakit": [
        ("gejala", "Seperti apa gejala {nama}?"),
        ("penyebab", "Apa penyebab {nama}?"),
        ("pengendalian", "Bagaimana cara mengendalikan {nama}?"),
        ("pencegahan", "Bagaimana cara mencegah {nama}?"),
    ],
    "hama": [
        ("gejala", "Seperti apa gejala serangan {nama}?"),
        ("pengendalian", "Bagaimana cara mengendalikan {nama}?"),
        ("pencegahan", "Bagaimana cara mencegah serangan {nama}?"),
        ("musuh_alami", "Apa saja musuh alami {nama}?"),
    ],
}

# Contoh pertanyaan umum untuk jawaban yang tidak menyebut hama atau penyakit apa pun
SARAN_UMUM = [
    "Penyakit padi apa saja yang disebabkan jamur?",
    "Apa gejala penyakit blas pada daun padi?",
    "Bagaimana cara mengendalikan wereng batang cokelat?",
]

POLA_DAFTAR = re.compile(
    r"\b(apa\s+saja|apa-apa\s+saja|sebutkan|jenis[\s-]+jenis|macam[\s-]+macam|daftar|berbagai)\b"
)


def _rapikan(teks: str) -> str:
    teks = (teks or "").lower().replace("coklat", "cokelat")
    return re.sub(r"\s+", " ", teks)


def cari_topik(teks: str) -> list[tuple[str, str]]:
    """Semua hama/penyakit yang disebut di teks, urut dari yang paling awal muncul."""
    teks = _rapikan(teks)
    temuan = []
    for nama, jenis, pola in _TOPIK:
        m = pola.search(teks)
        if m:
            temuan.append((m.start(), nama, jenis))
    return [(nama, jenis) for _, nama, jenis in sorted(temuan)]


def cari_aspek(teks: str) -> str | None:
    teks = _rapikan(teks)
    for aspek, pola in ASPEK:
        if pola.search(teks):
            return aspek
    return None


def _sama(a: str, b: str) -> bool:
    return re.sub(r"[^a-z0-9 ]", "", _rapikan(a)).strip() == re.sub(r"[^a-z0-9 ]", "", _rapikan(b)).strip()


def saran_lanjutan(pertanyaan: str, jawaban: str = "", maks: int = MAKS_SARAN) -> list[str]:
    """Daftar saran pertanyaan lanjutan (paling banyak `maks`)."""
    di_tanya = cari_topik(pertanyaan)
    di_jawab = cari_topik(jawaban)
    saran: list[str] = []

    if POLA_DAFTAR.search(_rapikan(pertanyaan)) and len(di_jawab) > 1:
        # Pertanyaan daftar: perdalam beberapa hama/penyakit yang disebut di jawaban
        for nama, jenis in di_jawab:
            saran.append(TEMPLAT[jenis][0][1].format(nama=nama))
    elif di_tanya or di_jawab:
        nama, jenis = (di_tanya or di_jawab)[0]
        aspek = cari_aspek(pertanyaan)
        for kode, templat in TEMPLAT[jenis]:
            if kode != aspek:
                saran.append(templat.format(nama=nama))
    else:
        saran = list(SARAN_UMUM)

    hasil: list[str] = []
    for s in saran:
        if not _sama(s, pertanyaan) and s not in hasil:
            hasil.append(s)
    return hasil[:maks]
