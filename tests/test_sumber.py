"""Uji tautan dan judul sumber rujukan (revisi seminar hasil)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_core import config, engine, sumber  # noqa: E402


def test_judul_dirapikan_dari_nama_berkas():
    assert sumber.judul_rapi("Blas/138_Pengendalian_penyakit_blas.pdf") == "Pengendalian penyakit blas"
    assert sumber.judul_rapi("Lain lain/M.Thamrin.pdf") == "M.Thamrin"


def test_tautan_menuju_halaman_yang_dikutip(monkeypatch):
    monkeypatch.setattr(config, "SUMBER_BASE_URL", "https://cdn.contoh/data_source")
    sumber._daftar.cache_clear()
    assert sumber.tautan("Lain lain/a b.pdf", 5) == "https://cdn.contoh/data_source/Lain%20lain/a%20b.pdf#page=5"
    assert sumber.tautan("-") is None


def test_tautan_resmi_dari_csv_didahulukan(monkeypatch, tmp_path):
    csv = tmp_path / "tautan.csv"
    csv.write_text("berkas,judul,tautan\nBlas/x.pdf,Pedoman Blas,https://doi.org/10.1/abc\n", encoding="utf-8")
    monkeypatch.setattr(sumber, "BERKAS_CSV", csv)
    sumber._daftar.cache_clear()
    r = sumber.rincian(2, {"source": "Blas/x.pdf", "page": 4})
    assert r == {"nomor": 2, "judul": "Pedoman Blas", "halaman": 4,
                 "tautan": "https://doi.org/10.1/abc", "berkas": "Blas/x.pdf"}
    sumber._daftar.cache_clear()


def test_csv_mencakup_semua_dokumen_sumber():
    pdf = {str(p.relative_to(ROOT / "data_source")).replace("\\", "/")
           for p in (ROOT / "data_source").rglob("*.pdf")}
    sumber._daftar.cache_clear()
    assert pdf and pdf <= set(sumber._daftar())


def test_hanya_sumber_yang_dikutip():
    ctx = [{"source": "A/1_a.pdf", "page": 1}, {"source": "B/2_b.pdf", "page": 9}]
    hasil = engine.sumber_dikutip_rinci("Isi [2].", ctx)
    assert [r["nomor"] for r in hasil] == [2] and hasil[0]["halaman"] == 9


def test_nomor_rujukan_diurutkan_ulang_agar_jumlah_dan_nomor_cocok():
    """Masalah setelah revisi 3: "Sumber rujukan (6)" tetapi nomornya sampai 8."""
    from rag_core import engine
    ctx = [{"source": f"D{i}.pdf", "page": 1, "text": ""} for i in range(1, 9)]
    ctx[6] = dict(ctx[2])   # potongan 7 berasal dari dokumen & halaman yang sama dengan 3
    jawaban, dikutip = engine.nomori_ulang_sitasi(
        "Blas [1][3][8]. Bercak [5, 6]. Hawar [7]. Tahun [2023].", ctx)
    assert jawaban == "Blas [1][2][3]. Bercak [4][5]. Hawar [2]. Tahun [2023]."
    assert [c["source"] for c in dikutip] == ["D1.pdf", "D3.pdf", "D8.pdf", "D5.pdf", "D6.pdf"]
    rinci = engine.sumber_dikutip_rinci(jawaban, dikutip)
    assert [r["nomor"] for r in rinci] == [1, 2, 3, 4, 5]


def test_tautan_resmi_pdf_dibuka_di_halaman_yang_dikutip(monkeypatch, tmp_path):
    csv = tmp_path / "tautan.csv"
    csv.write_text("berkas,judul,tautan\nUmum/a.pdf,A,https://contoh.org/a.pdf\n"
                   "Umum/b.pdf,B,https://doi.org/10.1/abc\n", encoding="utf-8")
    monkeypatch.setattr(sumber, "BERKAS_CSV", csv)
    sumber._daftar.cache_clear()
    assert sumber.tautan("Umum/a.pdf", 12) == "https://contoh.org/a.pdf#page=12"
    assert sumber.tautan("Umum/b.pdf", 12) == "https://doi.org/10.1/abc"
    sumber._daftar.cache_clear()
