"""Uji saran pertanyaan lanjutan.

Jalankan dari folder utama proyek:
    python -m pytest tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.saran import SARAN_UMUM, cari_aspek, cari_topik, saran_lanjutan  # noqa: E402


def test_aspek_lain_untuk_penyakit_yang_sama():
    hasil = saran_lanjutan("Apa gejala penyakit blas pada daun padi?")
    assert hasil == [
        "Apa penyebab penyakit blas?",
        "Bagaimana cara mengendalikan penyakit blas?",
        "Bagaimana cara mencegah penyakit blas?",
    ]


def test_hama_memakai_templat_hama():
    hasil = saran_lanjutan("Bagaimana cara mengendalikan wereng batang coklat?")
    assert "Bagaimana cara mengendalikan wereng batang cokelat?" not in hasil
    assert hasil[0] == "Seperti apa gejala serangan wereng batang cokelat?"
    assert "Apa saja musuh alami wereng batang cokelat?" in hasil


def test_nama_lain_dikenali():
    assert cari_topik("Apa itu beluk?") == [("penggerek batang padi", "hama")]
    assert cari_topik("daun mengering, apakah itu kresek?")[0][0].startswith("penyakit hawar daun bakteri")
    assert cari_topik("Pyricularia oryzae menyerang leher malai")[0][0] == "penyakit blas"


def test_nama_paling_awal_yang_dipakai():
    hasil = saran_lanjutan("Bagaimana cara mengendalikan wereng hijau penular tungro?")
    assert all("wereng hijau" in s for s in hasil)


def test_aspek_pencegahan_didahulukan():
    assert cari_aspek("Bagaimana cara mencegah serangan wereng?") == "pencegahan"
    assert cari_aspek("Kapan wereng cokelat perlu dikendalikan dengan insektisida?") == "pengendalian"


def test_kata_berimbuhan_dikenali():
    for q in ["Bagaimana cara mengendalikan keong mas di sawah?", "Apa pengendalian tikus?",
              "Cara membasmi walang sangit?", "Bagaimana mengatasi tungro?"]:
        assert cari_aspek(q) == "pengendalian", q
    hasil = saran_lanjutan("Bagaimana cara mengendalikan keong mas di sawah?")
    assert not any(s.startswith("Bagaimana cara mengendalikan") for s in hasil)


def test_gejala_tanpa_nama_memakai_topik_dari_jawaban():
    hasil = saran_lanjutan(
        "Daun padi mengering dari ujung dan pinggirnya, kenapa ya?",
        "Gejala tersebut sesuai dengan penyakit hawar daun bakteri atau kresek [1].",
    )
    assert hasil and all("hawar daun bakteri" in s for s in hasil)


def test_pertanyaan_daftar_memakai_topik_di_jawaban():
    hasil = saran_lanjutan(
        "Penyakit padi apa saja yang disebabkan oleh virus?",
        "- Tungro [1]\n- Kerdil rumput [2]\n- Kerdil hampa [2]",
    )
    assert hasil == [
        "Seperti apa gejala penyakit tungro?",
        "Seperti apa gejala penyakit kerdil rumput?",
        "Seperti apa gejala penyakit kerdil hampa?",
    ]


def test_di_luar_padi_diberi_contoh_umum():
    hasil = saran_lanjutan(
        "Bagaimana cara mengendalikan ulat pada jagung?",
        "Maaf, asisten ini khusus untuk hama dan penyakit tanaman padi.",
    )
    assert hasil == SARAN_UMUM


def test_tidak_mengulang_pertanyaan_dan_maksimal_tiga():
    for q in ["Apa penyebab penyakit tungro?", "Apa saja musuh alami tikus sawah?"]:
        hasil = saran_lanjutan(q)
        assert len(hasil) <= 3
        assert q not in hasil
