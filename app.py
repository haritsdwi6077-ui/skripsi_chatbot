"""Aplikasi web tanya jawab hama dan penyakit padi (Flask, di-deploy di Vercel).

Menggantikan app.py versi Streamlit. Aplikasi ini bersifat stateless: tidak ada
dokumen yang diproses saat runtime, hanya pencarian ke Qdrant Cloud dan
pemanggilan LLM.
"""

from flask import Flask, jsonify, render_template, request

from rag_core import config, engine

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "llm": config.LLM_MODEL,
            "embedding": config.EMBEDDING_MODEL,
            "collection": config.QDRANT_COLLECTION,
        }
    )


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    try:
        hasil = engine.answer(question)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Gagal memproses pertanyaan")
        return jsonify({"error": f"Sistem gagal memproses pertanyaan: {exc}"}), 500

    return jsonify(
        {
            "answer": hasil["answer"],
            "sources": engine.format_sources(hasil["contexts"]),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

