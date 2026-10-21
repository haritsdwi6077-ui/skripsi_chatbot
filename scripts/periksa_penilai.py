"""Periksa langkah demi langkah cara LLM penilai memberi skor pada satu soal.

Dipakai untuk mencari tahu mengapa factual correctness atau context recall bernilai
janggal (misalnya 0 padahal jawaban sama dengan rujukan). Skrip ini tidak mengubah
berkas CSV; ia hanya menampilkan klaim, putusan, dan alasan dari LLM penilai.

    python scripts/periksa_penilai.py --csv hasil\\uji_medium.csv --baris 17
    python scripts/periksa_penilai.py --csv hasil\\uji_medium.csv --baris 17 --metrik fc

Penilai yang dipakai sama dengan evaluate.py (JUDGE_* di .env).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Konsol Windows (cp1252) tidak bisa mencetak karakter seperti \u202f atau 【】 dari
# jawaban LLM, apalagi bila keluaran dialirkan ke Tee-Object. Paksa UTF-8.
for _aliran in (sys.stdout, sys.stderr):
    try:
        _aliran.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

AKAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AKAR))

# Pakai pengaturan penilai yang sama persis dengan evaluate.py
_spec = importlib.util.spec_from_file_location("evaluate_ragas", AKAR / "scripts" / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

from ragas.metrics import FactualCorrectness, LLMContextRecall  # noqa: E402
from ragas.metrics._context_recall import QCA  # noqa: E402


def buat_penilai():
    return ev.buat_penilai()  # pengaturan sama persis dengan evaluate.py


def garis(judul: str) -> None:
    print("\n" + "=" * 78 + f"\n{judul}\n" + "=" * 78)


async def periksa_fc(judge, response: str, reference: str) -> None:
    fc = FactualCorrectness(llm=judge)
    garis("FACTUAL CORRECTNESS")
    for nama, teks, pembanding, label in [
        ("JAWABAN SISTEM", response, reference, "didukung rujukan?"),
        ("JAWABAN RUJUKAN", reference, response, "ada di jawaban sistem?"),
    ]:
        klaim = await fc.decompose_claims(teks, None)
        print(f"\nKlaim yang dipecah dari {nama}: {len(klaim)} klaim")
        if not klaim:
            print("  (KOSONG: penilai tidak menghasilkan klaim, skor akan menjadi 0)")
            continue
        hasil = await fc.nli_prompt.generate(
            data=ev_nli_input(pembanding, klaim), llm=judge, callbacks=None)
        putusan = {s.statement: s for s in (hasil.statements or [])}
        if len(hasil.statements or []) != len(klaim):
            print(f"  PERHATIAN: {len(klaim)} klaim, tetapi {len(hasil.statements or [])} putusan")
        for i, k in enumerate(klaim, 1):
            s = putusan.get(k) or (hasil.statements[i - 1] if hasil.statements and i <= len(hasil.statements) else None)
            v = "YA " if s is not None and s.verdict else "TID"
            print(f"  {i}. [{v}] {k}")
            if s is not None:
                print(f"         {label} alasan: {s.reason}")
    skor = await fc.single_turn_ascore(_sampel(response=response, reference=reference))
    print(f"\nSkor factual correctness (F1) jika dihitung ulang sekarang: {skor}")


def ev_nli_input(premise: str, klaim: list[str]):
    from ragas.metrics._faithfulness import NLIStatementInput
    return NLIStatementInput(context=premise, statements=klaim)


def _sampel(**kw):
    from ragas.dataset_schema import SingleTurnSample
    return SingleTurnSample(**kw)


async def periksa_cr(judge, pertanyaan: str, konteks: list[str], reference: str) -> None:
    cr = LLMContextRecall(llm=judge)
    garis("CONTEXT RECALL")
    hasil = await cr.context_recall_prompt.generate(
        data=QCA(question=pertanyaan, context="\n".join(konteks), answer=reference),
        llm=judge, callbacks=None)
    daftar = hasil.classifications or []
    if not daftar:
        print("(KOSONG: penilai tidak mengklasifikasikan kalimat rujukan)")
    for i, c in enumerate(daftar, 1):
        print(f"  {i}. [{'ADA' if c.attributed else 'TIDAK'} di konteks] {c.statement}")
        print(f"         alasan: {c.reason}")
    if daftar:
        print(f"\nSkor context recall: {np.mean([1 if c.attributed else 0 for c in daftar]):.2f}")


async def utama() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--baris", type=int, required=True, help="nomor soal, mulai 1")
    p.add_argument("--metrik", default="fc,cr", help="fc, cr, atau fc,cr")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    r = df.iloc[a.baris - 1]
    pilih = {m.strip() for m in a.metrik.split(",")}
    print(f"Penilai: {ev.JUDGE_MODEL} | reasoning_effort: {ev.JUDGE_REASONING_EFFORT or '(bawaan)'}")
    print(f"Soal {a.baris}: {r['user_input']}")
    print(f"\nRUJUKAN : {r['reference']}")
    print(f"\nJAWABAN : {r['response']}")

    judge = buat_penilai()
    if "fc" in pilih:
        await periksa_fc(judge, r["response"], r["reference"])
    if "cr" in pilih and "retrieved_contexts" in df.columns and isinstance(r["retrieved_contexts"], str):
        await periksa_cr(judge, r["user_input"], list(ast.literal_eval(r["retrieved_contexts"])),
                         r["reference"])


if __name__ == "__main__":
    asyncio.run(utama())
