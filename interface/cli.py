"""Antarmuka baris perintah untuk pengujian cepat tanpa menjalankan web."""

from rag_core import config, engine


def run_cli_interface() -> None:
    print("\n=============================================")
    print("  CHATBOT RAG HAMA & PENYAKIT PADI (CLI)")
    print(f"  LLM: {config.LLM_MODEL} | Collection: {config.QDRANT_COLLECTION}")
    print("=============================================")

    while True:
        query = input("\nTanya sesuatu (ketik 'exit' untuk keluar): ").strip()
        if query.lower() in {"exit", "quit", "keluar"}:
            print("Keluar dari program. Semangat skripsinya!")
            break
        if not query:
            continue

        print("Mencari jawaban...")
        try:
            hasil = engine.answer(query)
        except Exception as exc:  # noqa: BLE001
            print(f"Gagal memproses pertanyaan: {exc}")
            continue

        print("\n--- Jawaban ---")
        print(hasil["answer"])
        print("\n[Sumber dokumen terkait]")
        for i, ctx in enumerate(hasil["contexts"], start=1):
            halaman = f" (hlm. {ctx['page']})" if ctx.get("page") else ""
            print(f"- {i}. {ctx['source']}{halaman}  skor={ctx['score']:.4f}")

