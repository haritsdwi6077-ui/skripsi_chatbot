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

Catatan: RAGAS memakai LLM sebagai penilai. Agar penilaian tidak bias, sebaiknya
LLM penilai berbeda dari LLM yang diuji (atur lewat JUDGE_* di bawah), dan LLM
penilai yang sama dipakai untuk seluruh model yang dibandingkan.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402
import pandas as pd  # noqa: E402

load_dotenv()

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
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


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", config.LLM_MODEL)
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", config.LLM_BASE_URL)
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", config.LLM_API_KEY)


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
    args = parser.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))

    model_uji = args.model or config.LLM_MODEL
    if JUDGE_MODEL == model_uji and JUDGE_BASE_URL == config.LLM_BASE_URL:
        print("PERINGATAN: LLM penilai sama dengan LLM yang diuji (menilai jawabannya sendiri).\n"
              "            Isi JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY di .env dengan model lain.\n")

    samples, waktu = [], []
    for i, item in enumerate(testset, start=1):
        if args.mode == "rag":
            hasil = engine.answer(item["question"], top_k=args.top_k, model=args.model)
        else:
            hasil = engine.answer_tanpa_rag(item["question"], model=args.model)
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

    judge = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, base_url=JUDGE_BASE_URL,
                   api_key=JUDGE_API_KEY, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(JinaEmbeddings())

    metrik_bersama = [
        ResponseRelevancy(llm=judge, embeddings=judge_embeddings),
        FactualCorrectness(llm=judge),
    ]
    if args.mode == "rag":
        daftar_metrik = [
            LLMContextPrecisionWithReference(llm=judge),
            LLMContextRecall(llm=judge),
            Faithfulness(llm=judge),
            *metrik_bersama,
        ]
    else:
        daftar_metrik = metrik_bersama
        for s in samples:
            s.pop("retrieved_contexts", None)  # mode llm tidak punya konteks

    hasil = evaluate(dataset=EvaluationDataset.from_list(samples), metrics=daftar_metrik)

    df = hasil.to_pandas()
    metrik = list(df.select_dtypes("number").columns)
    df = pd.concat([df.reset_index(drop=True), df_waktu.drop(columns=["user_input"])], axis=1)
    df.to_csv(args.out, index=False)
    print("\nRata-rata skor RAGAS:")
    print(df[metrik].mean().round(4).to_string())
    print(f"\nRincian per pertanyaan (skor + waktu) disimpan di {args.out}")
    print(f"Waktu per pertanyaan juga ada di {path_waktu}")


if __name__ == "__main__":
    main()

