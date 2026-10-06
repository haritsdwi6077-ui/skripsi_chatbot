"""Menjalankan chatbot melalui antarmuka baris perintah.

    python main.py

Basis pengetahuan harus sudah diunggah lebih dulu:

    python scripts/ingest.py --recreate
"""

from dotenv import load_dotenv

load_dotenv()

from database import vector_db  # noqa: E402
from interface.cli import run_cli_interface  # noqa: E402
from rag_core import config  # noqa: E402


def main() -> None:
    if not config.LLM_API_KEY:
        raise SystemExit("LLM_API_KEY belum diatur di file .env")
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY belum diatur di file .env")

    jumlah = vector_db.count()
    if jumlah == 0:
        raise SystemExit(
            "Collection masih kosong. Jalankan dulu: python scripts/ingest.py --recreate"
        )
    print(f"Basis pengetahuan siap: {jumlah} chunk.")
    run_cli_interface()


if __name__ == "__main__":
    main()

