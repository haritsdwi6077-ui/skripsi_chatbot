"""Mesin RAG: retrieval dari Qdrant Cloud, penyusunan prompt, lalu generation.

Versi sebelumnya memakai rantai LangChain dengan Gemini. Versi ini memanggil
API secara langsung agar dependensinya ringan (penting untuk deploy di Vercel)
dan agar model LLM mudah diganti untuk kebutuhan perbandingan antar-LLM.
"""

import time

from rag_core import config, embeddings, llm
from database import vector_db

SYSTEM_PROMPT = (
    "Anda adalah asisten penyuluh pertanian yang menjawab pertanyaan seputar "
    "hama dan penyakit tanaman padi. Jawablah HANYA berdasarkan konteks yang "
    "diberikan. Dokumen konteks dapat berbahasa Indonesia maupun Inggris, "
    "tetapi jawaban selalu ditulis dalam bahasa Indonesia yang ringkas dan "
    "mudah dipahami petani. Jika konteks tidak memuat informasi yang "
    "ditanyakan, katakan bahwa informasi tersebut tidak ditemukan pada basis "
    "pengetahuan dan jangan mengarang jawaban. Cantumkan nomor sumber, "
    "misalnya [1], pada bagian jawaban yang mengambil informasi dari konteks "
    "tersebut."
)

PROMPT_TEMPLATE = """Konteks:
{context}

Pertanyaan pengguna:
{question}

Jawaban:"""


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    vector = embeddings.embed_query(question)
    return vector_db.search(vector, top_k or config.TOP_K)


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        page = f", hlm. {ctx['page']}" if ctx.get("page") else ""
        blocks.append(f"[{i}] (Sumber: {ctx['source']}{page})\n{ctx['text']}")
    return PROMPT_TEMPLATE.format(
        context="\n\n".join(blocks) if blocks else "(tidak ada konteks)",
        question=question,
    )


def answer(question: str, top_k: int | None = None, model: str | None = None) -> dict:
    """Satu siklus penuh RAG: jawaban, konteks sumber, serta waktu tiap tahap.

    "timing" (detik): retrieval_s = embedding kueri + pencarian Qdrant,
    generation_s = panggilan LLM, total_s = keduanya ditambah penyusunan prompt.
    """
    t0 = time.perf_counter()
    contexts = retrieve(question, top_k=top_k)
    t1 = time.perf_counter()
    if not contexts:
        return {
            "answer": "Informasi tersebut tidak ditemukan pada basis pengetahuan.",
            "contexts": [],
            "timing": {"retrieval_s": round(t1 - t0, 3), "generation_s": 0.0,
                       "total_s": round(t1 - t0, 3)},
            "usage": {},
        }
    hasil = llm.generate_detail(
        build_prompt(question, contexts),
        system_prompt=SYSTEM_PROMPT,
        model=model,
    )
    t2 = time.perf_counter()
    return {
        "answer": hasil["text"],
        "contexts": contexts,
        "timing": {
            "retrieval_s": round(t1 - t0, 3),
            "generation_s": hasil["latency_s"],
            "total_s": round(t2 - t0, 3),
        },
        "usage": {k: hasil[k] for k in ("model", "prompt_tokens", "completion_tokens",
                                        "reasoning_tokens", "total_tokens")},
    }


def format_sources(contexts: list[dict]) -> list[str]:
    """Daftar sumber unik untuk ditampilkan di bawah jawaban."""
    hasil, sudah = [], set()
    for ctx in contexts:
        label = ctx["source"] + (f" hlm. {ctx['page']}" if ctx.get("page") else "")
        if label not in sudah:
            sudah.add(label)
            hasil.append(label)
    return hasil

