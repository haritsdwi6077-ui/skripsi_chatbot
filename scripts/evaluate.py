"""Evaluasi RAGAS: context precision, context recall, faithfulness, answer relevancy,
ditambah pencatatan waktu respons (retrieval, generation, total) dan jumlah token.

Dijalankan SECARA LOKAL:
    # LLM + RAG
    python scripts/evaluate.py --mode rag --out hasil_rag.csv
    # LLM saja (pembanding, tanpa retrieval)
    python scripts/evaluate.py --mode llm --out hasil_llm.csv

Metrik per mode:
    rag : context precision, context recall, faithfulness, answer relevancy,
          factual correctness
    llm : answer relevancy, factual correctness
Context precision/recall dan faithfulness membutuhkan konteks hasil retrieval,
sehingga hanya bisa dihitung pada mode rag. Answer relevancy dan factual
correctness dihitung pada kedua mode, jadi dua metrik inilah yang dipakai untuk
membandingkan LLM + RAG dengan LLM saja secara langsung.

Format testset.json:
    [{"question": "...", "ground_truth": "..."}, ...]

Setelah kedua mode selesai, rekap tabel BAB IV (statistik, uji Wilcoxon, waktu):
    python scripts/rekap_hasil.py --rag hasil_rag.csv --llm hasil_llm.csv

Uji coba cepat dengan 3 pertanyaan pertama:
    python scripts/evaluate.py --mode rag --out coba_rag.csv --batas 3

Menilai ulang hanya baris yang skornya masih kosong (tanpa menjawab ulang):
    python scripts/evaluate.py --mode rag --nilai-ulang hasil_rag.csv
Menilai ulang metrik tertentu pada soal tertentu dan mengganti skor lamanya:
    python scripts/evaluate.py --mode rag --nilai-ulang salinan.csv --metrik fc,cr --baris 2,12 --timpa

Catatan: RAGAS memakai LLM sebagai penilai. Agar penilaian tidak bias, sebaiknya
LLM penilai berbeda dari LLM yang diuji (atur lewat JUDGE_* di bawah), dan LLM
penilai yang sama dipakai untuk seluruh model yang dibandingkan.
"""

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402
import pandas as pd  # noqa: E402

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.metrics import (  # noqa: E402
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from rag_core import config, embeddings, engine  # noqa: E402


class JinaEmbeddings(Embeddings):
    """Embedding Jina untuk metrik ResponseRelevancy (membandingkan kemiripan teks)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        hasil = []
        for i in range(0, len(texts), 50):
            hasil.extend(embeddings.embed_texts(texts[i:i + 50], "text-matching"))
        return hasil

    def embed_query(self, text: str) -> list[float]:
        return embeddings.embed_texts([text], "text-matching")[0]


JUDGE_MODEL = os.environ.get("JUDGE_MODEL") or config.LLM_MODEL
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL") or config.LLM_BASE_URL
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY") or config.LLM_API_KEY
# Answer relevancy meminta beberapa jawaban sekaligus (parameter n). Banyak penyedia
# selain OpenAI (Groq, Gemini) tidak menerima n > 1, sehingga skornya kosong. Bila
# aktif, permintaan dikirim satu per satu. Matikan dengan JUDGE_BYPASS_N=0.
# Tingkat penalaran LLM penilai (none/low/medium/high), hanya untuk model bernalar
# seperti qwen-3.8-27b di Cerebras. Dikosongkan = bawaan penyedia. Nilai rendah
# menghemat token, karena token penalaran ikut dihitung dalam kuota harian.
JUDGE_REASONING_EFFORT = os.environ.get("JUDGE_REASONING_EFFORT", "").strip()
# Parameter tambahan khusus penyedia, dalam bentuk JSON, mis. untuk GLM di Z.ai:
# JUDGE_EXTRA_BODY={"thinking": {"type": "disabled"}}
JUDGE_EXTRA_BODY = os.environ.get("JUDGE_EXTRA_BODY", "").strip()
JUDGE_BYPASS_N = os.environ.get("JUDGE_BYPASS_N", "1").strip() not in ("0", "false", "False")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="data_source/testset.json")
    parser.add_argument("--out", default="hasil_evaluasi.csv")
    parser.add_argument("--model", default=None,
                        help="model LLM yang diuji (default: LLM_MODEL)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="paksa jumlah potongan (default: TOP_K, atau TOP_K_DAFTAR "
                             "untuk pertanyaan bertipe daftar)")
    parser.add_argument("--mode", choices=["rag", "llm"], default="rag",
                        help="rag = LLM + RAG (default), llm = LLM saja tanpa retrieval")
    parser.add_argument("--batas", type=int, default=0,
                        help="hanya jalankan N pertanyaan pertama (untuk uji coba), 0 = semua")
    parser.add_argument("--jeda", type=float, default=2.0,
                        help="jeda detik antarpertanyaan agar tidak terkena batas kuota API (default 2)")
    parser.add_argument("--workers", type=int, default=4,
                        help="jumlah permintaan penilaian RAGAS yang berjalan bersamaan (default 4)")
    parser.add_argument("--izinkan-penilai-sama", action="store_true",
                        help="tetap jalan walaupun LLM penilai sama dengan LLM yang diuji")
    parser.add_argument("--nilai-ulang", metavar="CSV", default=None,
                        help="nilai ulang baris berskor kosong pada CSV hasil sebelumnya, tanpa menjawab ulang")
    parser.add_argument("--metrik", default=None,
                        help="hanya metrik tertentu saat --nilai-ulang, mis. fc,cr "
                             "(cp=context precision, cr=context recall, f=faithfulness, "
                             "ar=answer relevancy, fc=factual correctness)")
    parser.add_argument("--baris", default=None,
                        help="hanya nomor soal tertentu saat --nilai-ulang, mis. 2,12,17 (mulai 1)")
    parser.add_argument("--timpa", action="store_true",
                        help="saat --nilai-ulang, ganti skor lama dengan skor baru (bukan hanya yang kosong)")
    parser.add_argument("--timeout", type=int, default=BATAS_WAKTU_BAWAAN,
                        help="batas waktu (detik) setiap tugas penilaian RAGAS (default 180); "
                             "naikkan bila penilai lambat, mis. 600")
    args = parser.parse_args()
    global BATAS_WAKTU
    BATAS_WAKTU = args.timeout

    model_uji = args.model or config.LLM_MODEL
    if JUDGE_MODEL == model_uji and JUDGE_BASE_URL == config.LLM_BASE_URL:
        pesan = ("LLM penilai sama dengan LLM yang diuji (model menilai jawabannya sendiri).\n"
                 "Isi JUDGE_MODEL, JUDGE_BASE_URL, dan JUDGE_API_KEY di .env dengan model lain.\n"
                 f"Berkas .env yang dibaca: {ENV_PATH} ({'ada' if ENV_PATH.exists() else 'TIDAK ADA'})\n"
                 f"JUDGE_MODEL yang terbaca: {os.environ.get('JUDGE_MODEL') or '(kosong)'}")
        if not args.izinkan_penilai_sama:
            raise SystemExit("BERHENTI: " + pesan + "\nTambahkan --izinkan-penilai-sama bila memang disengaja.")
        print("PERINGATAN: " + pesan + "\n")
    if args.nilai_ulang:
        pilih = None
        if args.metrik:
            pilih = {k.strip().lower() for k in args.metrik.split(",") if k.strip()}
            salah = pilih - set(KODE_METRIK)
            if salah:
                raise SystemExit(f"BERHENTI: kode metrik tidak dikenal: {', '.join(sorted(salah))}")
        nomor = [int(x) for x in args.baris.split(",") if x.strip()] if args.baris else None
        nilai_ulang(args.nilai_ulang, args.mode, args.workers, pilih, nomor, args.timpa)
        return

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    if args.batas:
        testset = testset[:args.batas]
    print(f"Model diuji: {model_uji} | penilai: {JUDGE_MODEL} | mode: {args.mode} | "
          f"{len(testset)} pertanyaan\n")

    def jawab(pertanyaan: str) -> dict:
        """Satu pertanyaan; gangguan jaringan atau kuota diulang maksimal 4 kali."""
        for ke in range(1, 5):
            try:
                if args.mode == "rag":
                    return engine.answer(pertanyaan, top_k=args.top_k, model=args.model)
                return engine.answer_tanpa_rag(pertanyaan, model=args.model)
            except Exception as exc:  # noqa: BLE001
                if ke == 4:
                    raise
                tunggu = 20 * ke  # batas token per menit Groq gratis cepat tercapai
                print(f"    gagal ({exc.__class__.__name__}: {str(exc)[:160]}), "
                      f"mencoba lagi dalam {tunggu} detik...")
                time.sleep(tunggu)
        raise RuntimeError("tidak terjangkau")

    samples, waktu = [], []
    for i, item in enumerate(testset, start=1):
        if i > 1 and args.jeda:
            time.sleep(args.jeda)
        hasil = jawab(item["question"])
        samples.append(
            {
                "user_input": item["question"],
                "retrieved_contexts": [c["text"] for c in hasil["contexts"]],
                "response": hasil["answer"],
                "reference": item["ground_truth"],
            }
        )
        w, u = hasil.get("timing", {}), hasil.get("usage", {})
        waktu.append({
            "retrieval_s": w.get("retrieval_s"), "generation_s": w.get("generation_s"),
            "total_s": w.get("total_s"), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": u.get("reasoning_tokens"),
        })
        print(f"  {i}/{len(testset)} pertanyaan dijawab ({w.get('total_s', 0):.2f} s)")

    # Simpan waktu lebih dulu, agar tetap ada walaupun tahap RAGAS gagal
    df_waktu = pd.DataFrame(waktu)
    df_waktu.insert(0, "user_input", [s["user_input"] for s in samples])
    df_waktu.insert(1, "model", model_uji)
    df_waktu.insert(2, "mode", args.mode)
    path_waktu = Path(args.out).with_name(Path(args.out).stem + "_waktu.csv")
    df_waktu.to_csv(path_waktu, index=False)
    ringkas = df_waktu[["retrieval_s", "generation_s", "total_s"]].astype(float)
    print(f"\nWaktu respons ({model_uji}, mode {args.mode}, n={len(df_waktu)}), detik:")
    print(pd.DataFrame({
        "rata-rata": ringkas.mean(), "median": ringkas.median(),
        "p95": ringkas.quantile(0.95), "min": ringkas.min(), "maks": ringkas.max(),
    }).round(3).to_string())

    df = nilai(samples, args.mode, args.workers)
    metrik = list(df.select_dtypes("number").columns)
    df = pd.concat([df.reset_index(drop=True), df_waktu.drop(columns=["user_input"])], axis=1)
    df.to_csv(args.out, index=False)
    laporkan(df, metrik, args.out, args.mode)
    print(f"Waktu per pertanyaan juga ada di {path_waktu}")


def buat_penilai() -> LangchainLLMWrapper:
    """LLM penilai RAGAS sesuai pengaturan JUDGE_* di .env."""
    opsi = {}
    if JUDGE_REASONING_EFFORT:
        opsi["reasoning_effort"] = JUDGE_REASONING_EFFORT
    if JUDGE_EXTRA_BODY:
        try:
            opsi["extra_body"] = json.loads(JUDGE_EXTRA_BODY)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"BERHENTI: JUDGE_EXTRA_BODY bukan JSON yang sah: {exc}")
    llm_penilai = ChatOpenAI(model=JUDGE_MODEL, base_url=JUDGE_BASE_URL,
                             api_key=JUDGE_API_KEY, temperature=0, **opsi)
    try:
        return LangchainLLMWrapper(llm_penilai, bypass_n=JUDGE_BYPASS_N)
    except TypeError:  # versi RAGAS lama belum mengenal bypass_n
        return LangchainLLMWrapper(llm_penilai)


BATAS_WAKTU = int(os.environ.get("RAGAS_TIMEOUT", "180"))  # detik per tugas; --timeout atau RAGAS_TIMEOUT
BATAS_WAKTU_BAWAAN = BATAS_WAKTU


KODE_METRIK = {  # kode singkat untuk --metrik -> awalan nama kolom RAGAS
    "cp": "llm_context_precision", "cr": "context_recall", "f": "faithfulness",
    "ar": "answer_relevancy", "fc": "factual_correctness",
}


def nilai(samples: list[dict], mode: str, workers: int, pilih: set | None = None) -> pd.DataFrame:
    """Hitung metrik RAGAS untuk daftar sampel; hasilnya satu baris per sampel.
    pilih: himpunan kode metrik (cp, cr, f, ar, fc); None = semua metrik mode ini."""
    judge = buat_penilai()
    judge_embeddings = LangchainEmbeddingsWrapper(JinaEmbeddings())

    metrik_bersama = [
        ("ar", ResponseRelevancy(llm=judge, embeddings=judge_embeddings)),
        ("fc", FactualCorrectness(llm=judge)),
    ]
    if mode == "rag":
        daftar_metrik = [
            ("cp", LLMContextPrecisionWithReference(llm=judge)),
            ("cr", LLMContextRecall(llm=judge)),
            ("f", Faithfulness(llm=judge)),
            *metrik_bersama,
        ]
    else:
        daftar_metrik = metrik_bersama
    daftar_metrik = [m for kode, m in daftar_metrik if pilih is None or kode in pilih]
    if not daftar_metrik:
        raise SystemExit("BERHENTI: tidak ada metrik yang cocok dengan --metrik untuk mode ini.")
    if mode != "rag":
        samples = [{k: v for k, v in s.items() if k != "retrieved_contexts"} for s in samples]

    hasil = evaluate(
        dataset=EvaluationDataset.from_list(samples), metrics=daftar_metrik,
        run_config=RunConfig(timeout=BATAS_WAKTU, max_retries=10, max_wait=60, max_workers=workers),
        raise_exceptions=False,
    )
    return hasil.to_pandas()


def laporkan(df: pd.DataFrame, metrik: list[str], out: str, mode: str) -> None:
    print("\nRata-rata skor RAGAS (skor kosong tidak dihitung):")
    print(df[metrik].mean().round(4).to_string())
    kosong = df[metrik].isna().sum()
    if kosong.any():
        print("\nJumlah skor kosong (NaN) per metrik, dari", len(df), "pertanyaan:")
        print(kosong.to_string())
        print("Skor kosong berarti LLM penilai gagal menghasilkan keluaran yang bisa diolah.\n"
              "Lihat pesan 'Exception raised in Job' di atas untuk penyebabnya, lalu nilai ulang\n"
              f"baris yang kosong dengan: python scripts/evaluate.py --mode {mode} --nilai-ulang {out}")
    else:
        print("\nSemua skor terisi.")
    print(f"\nRincian per pertanyaan disimpan di {out}")


def nilai_ulang(path: str, mode: str, workers: int, pilih: set | None = None,
                nomor: list[int] | None = None, timpa: bool = False) -> None:
    """Nilai ulang skor pada CSV hasil sebelumnya tanpa menjawab ulang pertanyaan.

    Bawaan: hanya sel yang kosong diisi, skor yang sudah ada tidak diubah.
    pilih : kode metrik yang dinilai ulang (None = semua).
    nomor : nomor soal (mulai 1) yang dinilai ulang (None = semua baris).
    timpa : True = skor lama pada metrik dan soal terpilih diganti dengan skor baru.
    """
    df = pd.read_csv(path)
    kolom_info = {"model", "mode", "retrieval_s", "generation_s", "total_s",
                  "prompt_tokens", "completion_tokens", "reasoning_tokens"}
    metrik = [c for c in df.select_dtypes("number").columns if c not in kolom_info]
    sasaran = [c for c in metrik
               if pilih is None or any(c.startswith(KODE_METRIK[k]) for k in pilih)]
    calon = [n - 1 for n in nomor if 1 <= n <= len(df)] if nomor else list(df.index)
    if timpa:
        baris = calon
    else:
        baris = [i for i in calon if df.loc[i, sasaran].isna().any()]
    print(f"Penilai: {JUDGE_MODEL} | metrik: {', '.join(sasaran)} | "
          f"{len(baris)} baris dinilai ulang{' (skor lama ditimpa)' if timpa else ''}")
    if not baris:
        return
    samples = []
    for i in baris:
        s = {"user_input": df.at[i, "user_input"], "response": df.at[i, "response"],
             "reference": df.at[i, "reference"]}
        if mode == "rag":
            s["retrieved_contexts"] = list(ast.literal_eval(df.at[i, "retrieved_contexts"]))
        samples.append(s)
    baru = nilai(samples, mode, workers, pilih)
    terisi = 0
    for j, i in enumerate(baris):
        for m in sasaran:
            if m not in baru.columns or pd.isna(baru.at[j, m]):
                continue
            if timpa or pd.isna(df.at[i, m]):
                df.at[i, m] = baru.at[j, m]
                terisi += 1
    df.to_csv(path, index=False)
    print(f"\n{terisi} skor {'diperbarui' if timpa else 'kosong berhasil diisi'}.")
    laporkan(df, metrik, path, mode)

if __name__ == "__main__":
    main()

