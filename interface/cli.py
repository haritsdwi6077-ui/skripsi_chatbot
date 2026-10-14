"""Antarmuka baris perintah untuk pengujian cepat tanpa menjalankan web."""

import re

from rag_core import config, engine


def rapikan_teks(teks: str) -> str:
    """Membersihkan sisa format yang tidak cocok untuk terminal."""
    teks = teks.replace("\u2011", "-").replace("\u00a0", " ")
    teks = re.sub(r"<br\s*/?>", "\n    ", teks, flags=re.IGNORECASE)
    teks = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", teks)
    return re.sub(r"\n{3,}", "\n\n", teks).strip()


def run_cli_interface(tanpa_rag: bool = False) -> None:
    mode = "LLM SAJA, TANPA RAG" if tanpa_rag else "RAG"
    print("\n=============================================")
    print(f"  CHATBOT HAMA & PENYAKIT PADI (CLI, mode {mode})")
    print(f"  LLM: {config.LLM_MODEL} | Collection: {config.QDRANT_COLLECTION}")
    print("=============================================")

    print("  Ketik 'baru' untuk memulai percakapan baru.")
    riwayat: list[dict] = []
    while True:
        query = input("\nTanya sesuatu (ketik 'exit' untuk keluar): ").strip()
        if query.lower() in {"exit", "quit", "keluar"}:
            print("Keluar dari program. Semangat skripsinya!")
            break
        if query.lower() == "baru":
            riwayat.clear()
            print("Percakapan baru dimulai.")
            continue
        if not query:
            continue

        print("Mencari jawaban...")
        try:
            if tanpa_rag:
                hasil = engine.answer_tanpa_rag(query, riwayat=riwayat)
            else:
                hasil = engine.answer(query, riwayat=riwayat)
        except Exception as exc:  # noqa: BLE001
            print(f"Gagal memproses pertanyaan: {exc}")
            continue

        riwayat.append({"question": query, "answer": hasil["answer"]})
        del riwayat[:-engine.MAKS_RIWAYAT]

        mandiri = hasil.get("pertanyaan_mandiri", query)
        if mandiri != query:
            print(f"(Ditafsirkan sebagai: {mandiri})")
        print("\n--- Jawaban ---")
        print(rapikan_teks(hasil["answer"]))
        if hasil["contexts"]:
            print("\n[Sumber dokumen terkait]")
        for i, ctx in enumerate(hasil["contexts"], start=1):
            halaman = f" (hlm. {ctx['page']})" if ctx.get("page") else ""
            print(f"- {i}. {ctx['source']}{halaman}  skor={ctx['score']:.4f}")
        w = hasil.get("timing", {})
        if w:
            if w.get("rewrite_s"):
                print(f"\n[Waktu] tulis ulang {w['rewrite_s']:.2f} s", end="")
            print(f"\n[Waktu] retrieval {w['retrieval_s']:.2f} s | "
                  f"generation {w['generation_s']:.2f} s | total {w['total_s']:.2f} s")

