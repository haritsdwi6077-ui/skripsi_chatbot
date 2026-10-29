"""Aplikasi web tanya jawab hama dan penyakit padi (Flask, di-deploy di Vercel).

Menggantikan app.py versi Streamlit. Aplikasi ini bersifat stateless: tidak ada
dokumen yang diproses saat runtime, hanya pencarian ke Qdrant Cloud dan
pemanggilan LLM.

Mode jawaban dipilih per pertanyaan lewat field "mode":
    "rag" (bawaan) : LLM + RAG, jawaban dari dokumen rujukan
    "llm"          : LLM saja tanpa retrieval (pembanding penelitian)
Pilihan mode di halaman web bisa disembunyikan dengan TAMPILKAN_MODE=0,
misalnya saat aplikasi dipakai petani dan bukan untuk demo penelitian.
"""

import os


from flask import Flask, jsonify, render_template, request

from rag_core import config, engine, saran

app = Flask(__name__)


TAMPILKAN_MODE = os.environ.get("TAMPILKAN_MODE", "1").strip() != "0"


@app.get("/")
def index():
    return render_template("index.html", tampilkan_mode=TAMPILKAN_MODE)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "llm": config.LLM_MODEL,
            "modes": ["rag", "llm"] if TAMPILKAN_MODE else ["rag"],
            "embedding": config.EMBEDDING_MODEL,
            "collection": config.QDRANT_COLLECTION,
            "gaya_jawaban": config.GAYA_JAWABAN,
        }
    )


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    riwayat = []
    for g in (data.get("history") or [])[-engine.MAKS_RIWAYAT:]:
        if isinstance(g, dict):
            riwayat.append({"question": str(g.get("question") or "")[:500],
                            "answer": str(g.get("answer") or "")})

    mode = "llm" if (data.get("mode") == "llm" and TAMPILKAN_MODE) else "rag"

    try:
        if mode == "llm":
            hasil = engine.answer_tanpa_rag(question, riwayat=riwayat)
        else:
            hasil = engine.answer(question, riwayat=riwayat)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Gagal memproses pertanyaan")
        return jsonify({"error": f"Sistem gagal memproses pertanyaan: {exc}"}), 500

    pertanyaan_mandiri = hasil.get("pertanyaan_mandiri", question)
    return jsonify(
        {
            "answer": hasil["answer"],
            "mode": mode,
            "sources": engine.sumber_dikutip(hasil["answer"], hasil["contexts"]),
            "standalone_question": pertanyaan_mandiri,
            "timing": hasil.get("timing", {}),
            # Saran pertanyaan lanjutan (aturan, tanpa LLM); lihat rag_core/saran.py
            "suggestions": saran.saran_lanjutan(pertanyaan_mandiri, hasil["answer"]),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

