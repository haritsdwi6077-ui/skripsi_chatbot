"""Evaluasi RAGAS: context precision, context recall, faithfulness, answer relevancy.

Dijalankan SECARA LOKAL:
    python scripts/evaluate.py --testset data_source/testset.json --out hasil_evaluasi.csv

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

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from rag_core import config, engine  # noqa: E402

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", config.LLM_MODEL)
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", config.LLM_BASE_URL)
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", config.LLM_API_KEY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="data_source/testset.json")
    parser.add_argument("--out", default="hasil_evaluasi.csv")
    parser.add_argument("--model", default=None,
                        help="model LLM yang diuji (default: LLM_MODEL)")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))

    samples = []
    for i, item in enumerate(testset, start=1):
        hasil = engine.answer(item["question"], top_k=args.top_k, model=args.model)
        samples.append(
            {
                "user_input": item["question"],
                "retrieved_contexts": [c["text"] for c in hasil["contexts"]],
                "response": hasil["answer"],
                "reference": item["ground_truth"],
            }
        )
        print(f"  {i}/{len(testset)} pertanyaan dijawab")

    judge = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, base_url=JUDGE_BASE_URL,
                   api_key=JUDGE_API_KEY, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=f"models/{config.EMBEDDING_MODEL}",
            google_api_key=config.GEMINI_API_KEY,
        )
    )

    hasil = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[
            LLMContextPrecisionWithReference(llm=judge),
            LLMContextRecall(llm=judge),
            Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=judge_embeddings),
        ],
    )

    df = hasil.to_pandas()
    df.to_csv(args.out, index=False)
    print("\nRata-rata skor:")
    print(df.select_dtypes("number").mean().round(4).to_string())
    print(f"\nRincian per pertanyaan disimpan di {args.out}")


if __name__ == "__main__":
    main()

