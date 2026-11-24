"""Uji mode percakapan aplikasi web (revisi seminar hasil), tanpa memanggil API.

Pemanggilan LLM dan retrieval diganti tiruan, sehingga uji ini menguji alur dan
cadangan, bukan kualitas jawaban model.

Jalankan dari folder utama proyek:
    python -m pytest tests
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_core import engine  # noqa: E402


def _hasil(teks: str) -> dict:
    return {"text": teks, "latency_s": 0.1, "model": "tiruan", "prompt_tokens": 1,
            "completion_tokens": 1, "reasoning_tokens": 0, "total_tokens": 2}


def _pasang(monkeypatch, balasan: dict, konteks=None):
    """balasan: system_prompt -> teks yang dikembalikan LLM tiruan."""
    panggilan = []

    def generate_detail(prompt, system_prompt="", **kw):
        panggilan.append({"prompt": prompt, "system": system_prompt, **kw})
        for kunci, teks in balasan.items():
            if system_prompt == kunci:
                if isinstance(teks, Exception):
                    raise teks
                return _hasil(teks)
        raise AssertionError("system prompt tak terduga")

    monkeypatch.setattr(engine.llm, "generate_detail", generate_detail)
    dicari = []
    monkeypatch.setattr(engine, "retrieve", lambda q, top_k=None: dicari.append(q) or (
        konteks if konteks is not None else
        [{"source": "Blas/a.pdf", "page": 3, "text": "Gejala blas: bercak belah ketupat."}]))
    return panggilan, dicari


def test_pertanyaan_singkat_dipahami_sebelum_dicari(monkeypatch):
    panggilan, dicari = _pasang(monkeypatch, {
        engine.SYSTEM_PROMPT_PAHAMI: '{"jenis": "pertanyaan", "kueri": "Apa gejala penyakit blas pada padi?"}',
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Bercaknya berbentuk belah ketupat [1].",
    })
    hasil = engine.answer("gejala blas gmn", percakapan=True)
    assert dicari == ["Apa gejala penyakit blas pada padi?"]
    assert hasil["jenis"] == "pertanyaan"
    assert hasil["pertanyaan_mandiri"] == "Apa gejala penyakit blas pada padi?"
    # Pesan asli pengguna tetap dikirim ke LLM, bukan hanya hasil tafsirannya
    assert "gejala blas gmn" in panggilan[-1]["prompt"]


def test_salam_tidak_mencari_dokumen(monkeypatch):
    _, dicari = _pasang(monkeypatch, {
        engine.SYSTEM_PROMPT_PAHAMI: '```json\n{"jenis": "sapaan", "kueri": ""}\n```',
        engine.SYSTEM_PROMPT_SAPAAN: "Halo! Silakan tanya soal padi Anda.",
    })
    hasil = engine.answer("halo", percakapan=True)
    assert dicari == []
    assert hasil["jenis"] == "sapaan" and hasil["contexts"] == []


def test_cadangan_bila_llm_pemaham_gagal(monkeypatch):
    _, dicari = _pasang(monkeypatch, {
        engine.SYSTEM_PROMPT_PAHAMI: "bukan json",
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Jawaban [1].",
    })
    engine.answer("padi adalah", percakapan=True)
    assert dicari == ["Apa itu padi?"]   # kembali ke lengkapi_pertanyaan()


def test_riwayat_ikut_dikirim_ke_llm(monkeypatch):
    panggilan, _ = _pasang(monkeypatch, {
        engine.SYSTEM_PROMPT_PAHAMI: '{"jenis": "pertanyaan", "kueri": "Bagaimana cara mencegah penyakit blas?"}',
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Gunakan varietas tahan [1].",
    })
    riwayat = [{"question": "blas itu apa", "answer": "Blas adalah penyakit jamur."}]
    engine.answer("cegahnya gmn", riwayat=riwayat, percakapan=True)
    assert "blas itu apa" in panggilan[0]["prompt"]
    assert panggilan[-1]["riwayat"] == riwayat


def test_jalur_evaluasi_tidak_berubah(monkeypatch):
    """Tanpa percakapan=True, answer() memakai prompt lama yang dievaluasi di skripsi."""
    panggilan, dicari = _pasang(monkeypatch, {
        engine.system_prompt_rag(): "Jawaban [1].",
    })
    engine.answer("Apa gejala penyakit blas?")
    assert dicari == ["Apa gejala penyakit blas?"]
    assert len(panggilan) == 1 and panggilan[0]["system"] == engine.system_prompt_rag()
