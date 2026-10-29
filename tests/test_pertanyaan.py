"""Uji pelengkap pertanyaan yang terpotong.

Jalankan dari folder utama proyek:
    python -m pytest tests
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_core.pertanyaan import lengkapi_pertanyaan  # noqa: E402


def test_pola_pengertian():
    assert lengkapi_pertanyaan("PADI ADALAH") == "Apa itu padi?"
    assert lengkapi_pertanyaan("padi adalah") == "Apa itu padi?"
    assert lengkapi_pertanyaan("Blas itu?") == "Apa itu blas?"
    assert lengkapi_pertanyaan("pengertian tungro") == "Apa itu tungro?"
    assert lengkapi_pertanyaan("Definisi dari hama terpadu") == "Apa itu hama terpadu?"


def test_nama_hama_atau_penyakit_saja():
    assert lengkapi_pertanyaan("wereng cokelat") == "Apa itu wereng cokelat?"
    assert lengkapi_pertanyaan("HDB") == "Apa itu HDB?"
    assert lengkapi_pertanyaan("HDB ADALAH") == "Apa itu HDB?"


def test_pertanyaan_lengkap_tidak_diubah():
    for q in ["Apa gejala penyakit blas?", "cara pencegahannya bagaimana?",
              "Daun padi menguning dari ujung, kenapa?", "daun menguning"]:
        assert lengkapi_pertanyaan(q) == q
    assert lengkapi_pertanyaan("APA GEJALA BLAS?") == "Apa gejala blas?"


def test_semua_pertanyaan_test_set_tidak_berubah():
    """Menjamin hasil evaluasi RAGAS tetap sama dengan kode ini."""
    testset = json.loads((ROOT / "data_source" / "testset.json").read_text(encoding="utf-8"))
    berubah = [x["question"] for x in testset if lengkapi_pertanyaan(x["question"]) != x["question"]]
    assert berubah == []
