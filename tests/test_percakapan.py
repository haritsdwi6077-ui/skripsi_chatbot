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

import pytest  # noqa: E402

from rag_core import engine, simpanan  # noqa: E402


@pytest.fixture(autouse=True)
def _simpanan_bersih(monkeypatch, tmp_path):
    """Setiap uji mulai tanpa jawaban tersimpan, tanpa berkas, dan tanpa Redis."""
    monkeypatch.setattr(simpanan, "BERKAS_JAWABAN", tmp_path / "tidak_ada.json")
    monkeypatch.setattr(engine.config, "REDIS_URL", "")
    simpanan.kosongkan_memori()
    yield
    simpanan.kosongkan_memori()


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
        engine.SYSTEM_PROMPT_PAHAMI: '{"jenis": "pertanyaan", "aspek": "lainnya", "kueri": "Penyakit apa yang menyebabkan bercak belah ketupat pada daun padi?"}',
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Kemungkinan blas [1].",
    })
    hasil = engine.answer("daun sy ada bercak kek belah ketupat", percakapan=True)
    assert dicari == ["Penyakit apa yang menyebabkan bercak belah ketupat pada daun padi?"]
    assert hasil["jenis"] == "pertanyaan"
    # Pesan asli pengguna tetap dikirim ke LLM, bukan hanya hasil tafsirannya
    assert "daun sy ada bercak kek belah ketupat" in panggilan[-1]["prompt"]


def test_pertanyaan_umum_dipahami_dengan_aturan_tanpa_llm(monkeypatch):
    """Kalimat sederhana dipahami dengan aturan tetap, jadi tidak bergantung tebakan LLM."""
    panggilan, dicari = _pasang(monkeypatch, {
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Bercaknya berbentuk belah ketupat [1].",
    })
    engine.answer("gejala blas gmn", percakapan=True)
    assert dicari == ["Apa gejala penyakit blas pada tanaman padi?"]
    assert all(p["system"] != engine.SYSTEM_PROMPT_PAHAMI for p in panggilan)


@pytest.mark.parametrize("pesan, kueri", [
    ("APA PENYAKIT PADI", engine.KUERI_DAFTAR["daftar_penyakit"]),
    ("apa penyakit pada padi", engine.KUERI_DAFTAR["daftar_penyakit"]),
    ("penyakit padi apa aja", engine.KUERI_DAFTAR["daftar_penyakit"]),
    ("sebutkan jenis-jenis penyakit pada tanaman padi", engine.KUERI_DAFTAR["daftar_penyakit"]),
    ("Apa saja penyakit yang menyerang padi?", engine.KUERI_DAFTAR["daftar_penyakit"]),
    ("hama padi apa aja", engine.KUERI_DAFTAR["daftar_hama"]),
    ("apa saja hama dan penyakit padi", engine.KUERI_DAFTAR["daftar_hama_penyakit"]),
    ("wbc gmn ngatasinya", "Bagaimana cara mengendalikan wereng batang cokelat pada tanaman padi?"),
    ("obat wereng coklat apa", "Bagaimana cara mengendalikan wereng batang cokelat pada tanaman padi?"),
    ("blas itu apa", "Apa itu penyakit blas pada tanaman padi?"),
    ("bercak coklat gejalanya seperti apa", "Apa gejala penyakit bercak cokelat pada tanaman padi?"),
    ("kenapa padi kena tungro", "Apa penyebab penyakit tungro pada tanaman padi?"),
    ("tikus sawah gmn ngusirnya", "Bagaimana cara mengendalikan tikus sawah pada tanaman padi?"),
])
def test_aturan_mengenali_kalimat_sehari_hari(pesan, kueri):
    assert engine.pahami_aturan(pesan) == kueri


@pytest.mark.parametrize("pesan", [
    "halo", "obatnya apa", "padi saya daunnya kuning", "penyakit padi yang disebabkan jamur",
    "blas dan tungro bedanya apa", "apakah blas menyerang jagung", "musuh alami blas",
])
def test_aturan_menyerahkan_kalimat_lain_ke_llm(pesan):
    assert engine.pahami_aturan(pesan) is None


def test_tulis_ulang_llm_yang_sederhana_tetap_dibakukan(monkeypatch):
    """LLM salah memilih aspek, tetapi kalimat tulis ulangnya dibakukan oleh aturan."""
    _, dicari = _pasang(monkeypatch, {
        engine.SYSTEM_PROMPT_PAHAMI: '{"jenis":"pertanyaan","aspek":"lainnya","topik":"","kueri":"Apa saja penyakit yang menyerang padi?"}',
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Blas [1].",
    })
    engine.answer("penyakit yg sering nyerang padi kita apa ya", percakapan=True)
    assert dicari == [engine.KUERI_DAFTAR["daftar_penyakit"]]


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
        engine.SYSTEM_PROMPT_PAHAMI: '{"jenis": "pertanyaan", "kueri": "Apakah penyakit blas bisa menular lewat benih?"}',
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Gunakan varietas tahan [1].",
    })
    riwayat = [{"question": "blas itu apa", "answer": "Blas adalah penyakit jamur."}]
    engine.answer("cegahnya gmn", riwayat=riwayat, percakapan=True)
    assert "blas itu apa" in panggilan[0]["prompt"]
    assert panggilan[-1]["riwayat"] == riwayat


def test_kalimat_berbeda_arti_sama_dicari_dengan_kueri_yang_sama(monkeypatch):
    """Masalah setelah seminar: susunan kata berbeda menghasilkan jawaban berbeda."""
    tafsiran_llm = {
        # Tafsiran LLM sengaja dibuat berbeda-beda susunan katanya
        "wereng coklat gmn ngatasinya": '{"jenis":"pertanyaan","aspek":"pengendalian","topik":"wereng coklat","kueri":"Bagaimana mengatasi wereng coklat?"}',
        "cara basmi WBC": '{"jenis":"pertanyaan","aspek":"pengendalian","topik":"WBC","kueri":"Apa cara membasmi WBC di sawah?"}',
        "obat buat wereng batang cokelat apa ya": '{"jenis":"pertanyaan","aspek":"pengendalian","topik":"wereng batang cokelat","kueri":"Obat apa untuk wereng batang cokelat?"}',
    }
    dicari = []

    def generate_detail(prompt, system_prompt="", **kw):
        if system_prompt == engine.SYSTEM_PROMPT_PAHAMI:
            pesan = prompt.rsplit("Pesan pengguna: ", 1)[1]
            return _hasil(tafsiran_llm[pesan])
        return _hasil("Jawaban [1].")

    monkeypatch.setattr(engine.llm, "generate_detail", generate_detail)
    monkeypatch.setattr(engine, "retrieve", lambda q, top_k=None: dicari.append(q) or
                        [{"source": "WBC/a.pdf", "page": 1, "text": "..."}])
    for pesan in tafsiran_llm:
        engine.answer(pesan, percakapan=True)
    assert set(dicari) == {"Bagaimana cara mengendalikan wereng batang cokelat pada tanaman padi?"}


def test_kueri_baku():
    assert engine.kueri_baku("gejala", "blast", "x") == "Apa gejala penyakit blas pada tanaman padi?"
    assert engine.kueri_baku("gejala", "", "ciri kena blas apa") == "Apa gejala penyakit blas pada tanaman padi?"
    assert engine.kueri_baku("pencegahan", "keong mas", "x") == \
        "Bagaimana cara mencegah serangan keong mas pada tanaman padi?"
    # Tidak bisa dibakukan: kueri LLM dipakai apa adanya
    assert engine.kueri_baku("lainnya", "", "Apa penyebab daun padi menguning?") == \
        "Apa penyebab daun padi menguning?"
    assert engine.kueri_baku("gejala", "blas dan tungro", "Apa beda gejala blas dan tungro?") == \
        "Apa beda gejala blas dan tungro?"
    assert engine.kueri_baku("pengendalian", "ulat grayak", "Cara mengendalikan ulat grayak?") == \
        "Cara mengendalikan ulat grayak?"


def test_jalur_evaluasi_tidak_berubah(monkeypatch):
    """Tanpa percakapan=True, answer() memakai prompt lama yang dievaluasi di skripsi."""
    panggilan, dicari = _pasang(monkeypatch, {
        engine.system_prompt_rag(): "Jawaban [1].",
    })
    engine.answer("Apa gejala penyakit blas?")
    assert dicari == ["Apa gejala penyakit blas?"]
    assert len(panggilan) == 1 and panggilan[0]["system"] == engine.system_prompt_rag()


def test_pertanyaan_daftar_dijawab_sama_walau_kalimat_dan_riwayat_berbeda(monkeypatch):
    """Masalah setelah revisi 2: "apa penyakit pada padi" menghasilkan rujukan dan
    susunan jawaban yang berbeda dari pertanyaan sebelumnya yang maknanya sama."""
    tafsiran_llm = {
        "penyakit padi apa aja": '{"jenis":"pertanyaan","aspek":"daftar_penyakit","topik":"","kueri":"Penyakit apa saja pada padi?"}',
        "apa penyakit pada padi": '{"jenis":"pertanyaan","aspek":"daftar_penyakit","topik":"","kueri":"Apa saja penyakit yang menyerang padi?"}',
    }
    dicari, dijawab = [], []
    nomor = iter(["Blas [1][3][6].", "Blas [1][4]."])   # LLM tiruan yang tidak konsisten

    def generate_detail(prompt, system_prompt="", **kw):
        if system_prompt == engine.SYSTEM_PROMPT_PAHAMI:
            return _hasil(tafsiran_llm[prompt.rsplit("Pesan pengguna: ", 1)[1]])
        dijawab.append(kw)
        return _hasil(next(nomor))

    monkeypatch.setattr(engine.llm, "generate_detail", generate_detail)
    monkeypatch.setattr(engine, "retrieve", lambda q, top_k=None: dicari.append(q) or
                        [{"source": "Blas/a.pdf", "page": 1, "text": "..."}] * 6)
    a = engine.answer("penyakit padi apa aja", percakapan=True)
    b = engine.answer("apa penyakit pada padi", percakapan=True,
                      riwayat=[{"question": "penyakit padi apa aja", "answer": a["answer"]}])
    assert dicari == ["Apa saja penyakit yang menyerang tanaman padi?"]   # dicari sekali saja
    assert dijawab[0]["riwayat"] == [] and dijawab[0]["seed"] == engine.config.LLM_SEED
    assert b["answer"] == a["answer"] and b["dari_cache"] is True


def test_kueri_bebas_tidak_disimpan(monkeypatch):
    _pasang(monkeypatch, {
        engine.SYSTEM_PROMPT_PAHAMI: '{"jenis":"pertanyaan","aspek":"lainnya","topik":"","kueri":"Apa penyebab daun padi menguning?"}',
        engine.SYSTEM_PROMPT_PERCAKAPAN: "Kemungkinan tungro [1].",
    })
    hasil = engine.answer("daun pd kuning knp", percakapan=True)
    assert hasil["dari_cache"] is False and not simpanan._MEMORI


def test_jawaban_dari_berkas_jawaban_baku_dipakai_lebih_dulu(monkeypatch, tmp_path):
    """Jawaban baku yang dibuat sekali (scripts/buat_jawaban_baku.py) selalu sama,
    di server mana pun dan setelah server dinyalakan ulang."""
    import json
    berkas = tmp_path / "jawaban_baku.json"
    berkas.write_text(json.dumps({"jawaban": {engine.KUERI_DAFTAR["daftar_penyakit"]: {
        "answer": "Blas [1].", "contexts": [{"source": "Blas/a.pdf", "page": 1, "text": "..."}]}}}))
    monkeypatch.setattr(simpanan, "BERKAS_JAWABAN", berkas)
    simpanan.kosongkan_memori()
    panggilan, dicari = _pasang(monkeypatch, {})
    for pesan in ("APA PENYAKIT PADI", "penyakit padi apa aja"):
        hasil = engine.answer(pesan, percakapan=True)
        assert hasil["answer"] == "Blas [1]." and hasil["dari_cache"] is True
    assert dicari == [] and panggilan == []


def test_redis_dipakai_bersama_dan_gagalnya_tidak_menggagalkan(monkeypatch):
    toko = {}

    def redis(perintah):
        if perintah[0] == "SET":
            toko[perintah[1]] = perintah[2]
            return "OK"
        return toko.get(perintah[1])

    monkeypatch.setattr(simpanan, "_redis", redis)
    _, dicari = _pasang(monkeypatch, {engine.SYSTEM_PROMPT_PERCAKAPAN: "Blas [1]."})
    engine.answer("apa penyakit padi", percakapan=True)
    simpanan.kosongkan_memori()          # seolah-olah server lain / server baru
    hasil = engine.answer("sebutkan penyakit padi", percakapan=True)
    assert hasil["dari_cache"] is True and len(dicari) == 1



def test_redis_mati_tidak_menggagalkan_jawaban(monkeypatch):
    import requests

    def gagal(*a, **kw):
        raise requests.ConnectionError("Redis mati")

    monkeypatch.setattr(engine.config, "REDIS_URL", "https://contoh.invalid")
    monkeypatch.setattr(engine.config, "REDIS_TOKEN", "x")
    monkeypatch.setattr(simpanan.requests, "post", gagal)
    _pasang(monkeypatch, {engine.SYSTEM_PROMPT_PERCAKAPAN: "Blas [1]."})
    assert engine.answer("apa penyakit padi", percakapan=True)["answer"] == "Blas [1]."


def test_setiap_kueri_baku_dikenali_ulang_oleh_aturan():
    """Dipakai scripts/buat_jawaban_baku.py: kueri baku yang dikirim ulang tetap kueri itu."""
    for kueri in engine.semua_kueri_baku():
        assert engine.pahami_aturan(kueri) == kueri
