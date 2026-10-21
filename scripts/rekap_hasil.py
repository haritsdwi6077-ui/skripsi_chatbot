"""Rekap hasil evaluasi untuk tabel BAB IV (Tabel 4.18 sampai Tabel 4.22).

Jalankan setelah kedua mode dievaluasi dengan testset yang sama:

    python scripts/evaluate.py --mode rag --out hasil_rag.csv
    python scripts/evaluate.py --mode llm --out hasil_llm.csv
    python scripts/rekap_hasil.py --rag hasil_rag.csv --llm hasil_llm.csv

Keluaran ditampilkan di layar dan disimpan ke rekap_hasil.md. Skor yang kosong
(NaN, biasanya karena LLM penilai gagal mengurai jawaban) TIDAK dihitung sebagai
nol. Jumlah skor yang valid dilaporkan di kolom n agar terlihat jelas.

Butuh: pandas, scipy  (pip install scipy)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

METRIK_RAG = {
    "llm_context_precision_with_reference": "Context precision",
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "factual_correctness(mode=f1)": "Factual correctness",
}
METRIK_BERSAMA = ["answer_relevancy", "factual_correctness(mode=f1)"]
WAKTU = {"retrieval_s": "Retrieval", "generation_s": "Pembangkitan", "total_s": "Total"}


def seragamkan(df: pd.DataFrame) -> pd.DataFrame:
    """Nama kolom RAGAS berbeda antarversi, misalnya "factual_correctness" (0.4)
    dan "factual_correctness(mode=f1)" (0.2), jadi diseragamkan lebih dulu."""
    ganti = {}
    for c in df.columns:
        if c.startswith("factual_correctness"):
            ganti[c] = "factual_correctness(mode=f1)"
        elif c.startswith("llm_context_precision") or c == "context_precision":
            ganti[c] = "llm_context_precision_with_reference"
    return df.rename(columns=ganti)


def angka(x: float, d: int = 4) -> str:
    """Format angka gaya Indonesia: koma desimal."""
    if pd.isna(x):
        return "-"
    return f"{x:.{d}f}".replace(".", ",")


def statistik(df: pd.DataFrame, kolom: list[str]) -> list[list[str]]:
    baris = []
    for k in kolom:
        if k not in df.columns:
            continue
        s = pd.to_numeric(df[k], errors="coerce").dropna()
        baris.append([METRIK_RAG[k], str(len(s)), angka(s.mean()), angka(s.std(ddof=1)),
                      angka(s.min()), angka(s.max())])
    return baris


def tabel_md(judul: str, header: list[str], baris: list[list[str]]) -> str:
    teks = [f"\n### {judul}\n", "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header)]
    teks += ["| " + " | ".join(b) + " |" for b in baris]
    return "\n".join(teks) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rag", default="hasil_rag.csv")
    p.add_argument("--llm", default="hasil_llm.csv")
    p.add_argument("--testset", default="data_source/testset.json",
                   help="dipakai untuk rekap per jenis pertanyaan bila ada kunci 'jenis'")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", default="rekap_hasil.md")
    a = p.parse_args()

    rag = seragamkan(pd.read_csv(a.rag))
    llm = seragamkan(pd.read_csv(a.llm)) if Path(a.llm).exists() else None
    keluaran = [f"# Rekap hasil evaluasi\n\nBerkas: {a.rag}" + (f", {a.llm}" if llm is not None else "")]
    keluaran.append(f"\nJumlah pertanyaan: mode RAG = {len(rag)}"
                    + (f", mode tanpa RAG = {len(llm)}" if llm is not None else ""))

    head = ["Metrik", "n", "Rata-rata", "Simpangan baku", "Minimum", "Maksimum"]
    keluaran.append(tabel_md("Tabel 4.18 Hasil evaluasi RAGAS mode LLM dengan RAG", head,
                             statistik(rag, list(METRIK_RAG))))
    if llm is not None:
        keluaran.append(tabel_md("Tabel 4.19 Hasil evaluasi RAGAS mode LLM tanpa RAG", head,
                                 statistik(llm, METRIK_BERSAMA)))

        # Uji peringkat bertanda Wilcoxon, berpasangan per pertanyaan
        gabung = rag.merge(llm, on="user_input", suffixes=("_rag", "_llm"))
        baris = []
        for k in METRIK_BERSAMA:
            pasangan = gabung[[f"{k}_rag", f"{k}_llm"]].apply(pd.to_numeric, errors="coerce").dropna()
            x, y = pasangan[f"{k}_rag"], pasangan[f"{k}_llm"]
            beda = x - y
            naik, turun, sama = int((beda > 0).sum()), int((beda < 0).sum()), int((beda == 0).sum())
            if (beda != 0).sum() >= 1:
                uji = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                stat, nilai_p = uji.statistic, uji.pvalue
                keputusan = "H0 ditolak" if nilai_p < a.alpha else "H0 diterima"
            else:
                stat, nilai_p, keputusan = float("nan"), float("nan"), "tidak dapat diuji"
            baris.append([METRIK_RAG[k], str(len(pasangan)), angka(x.mean()), angka(y.mean()),
                          angka(x.mean() - y.mean()), angka(stat, 1), angka(nilai_p),
                          keputusan, f"{naik}/{turun}/{sama}"])
        keluaran.append(tabel_md(
            "Tabel 4.20 Perbandingan skor dan hasil uji Wilcoxon",
            ["Metrik", "n pasangan", "Dengan RAG", "Tanpa RAG", "Selisih", "Statistik T",
             "Nilai p", "Keputusan", "RAG lebih tinggi/lebih rendah/sama"], baris))

    # Waktu respons
    baris = []
    for nama, df in (("LLM dengan RAG", rag), ("LLM tanpa RAG", llm)):
        if df is None:
            continue
        for k, label in WAKTU.items():
            if nama == "LLM tanpa RAG" and k == "retrieval_s":
                continue
            s = pd.to_numeric(df[k], errors="coerce").dropna()
            baris.append([nama, label, angka(s.mean(), 3), angka(s.median(), 3),
                          angka(s.quantile(0.95), 3), angka(s.max(), 3)])
    keluaran.append(tabel_md("Tabel 4.22 Waktu respons kedua mode (detik)",
                             ["Mode", "Tahap", "Rata-rata", "Median", "Persentil ke-95", "Maksimum"],
                             baris))
    baris = []
    for nama, df in (("LLM dengan RAG", rag), ("LLM tanpa RAG", llm)):
        if df is None:
            continue
        baris.append([nama] + [angka(pd.to_numeric(df[k], errors="coerce").mean(), 0)
                               for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")])
    keluaran.append(tabel_md("Rata-rata pemakaian token per pertanyaan",
                             ["Mode", "Token masukan", "Token keluaran", "Token penalaran"], baris))

    # Rekap per jenis pertanyaan (bila testset memuat kunci "jenis")
    if Path(a.testset).exists():
        ts = json.loads(Path(a.testset).read_text(encoding="utf-8"))
        jenis = {t["question"]: t.get("jenis") for t in ts}
        if any(jenis.values()):
            baris = []
            rag["jenis"] = rag["user_input"].map(jenis)
            if llm is not None:
                llm["jenis"] = llm["user_input"].map(jenis)
            for j in sorted(x for x in set(jenis.values()) if x):
                r = rag[rag["jenis"] == j]
                b = [j, str(len(r))] + [angka(pd.to_numeric(r[k], errors="coerce").mean())
                                        for k in METRIK_RAG]
                if llm is not None:
                    l_ = llm[llm["jenis"] == j]
                    b += [angka(pd.to_numeric(l_[k], errors="coerce").mean()) for k in METRIK_BERSAMA]
                baris.append(b)
            head = ["Jenis", "n", "CP", "CR", "F", "AR (RAG)", "FC (RAG)"]
            if llm is not None:
                head += ["AR (tanpa RAG)", "FC (tanpa RAG)"]
            keluaran.append(tabel_md("Tabel 4.21 Rata-rata skor per jenis pertanyaan", head, baris))

    # Pertanyaan dengan skor kosong, untuk dicek ulang
    kosong = []
    for nama, df, kol in (("rag", rag, list(METRIK_RAG)), ("llm", llm, METRIK_BERSAMA)):
        if df is None:
            continue
        for _, r in df.iterrows():
            hilang = [METRIK_RAG[k] for k in kol if k in df.columns and pd.isna(r[k])]
            if hilang:
                kosong.append(f"- [{nama}] {r['user_input']}: {', '.join(hilang)}")
    if kosong:
        keluaran.append("\n### Skor kosong (perlu dijalankan ulang atau dicatat sebagai keterbatasan)\n")
        keluaran += kosong

    teks = "\n".join(keluaran)
    Path(a.out).write_text(teks + "\n", encoding="utf-8")
    print(teks)
    print(f"\nDisimpan ke {a.out}")


if __name__ == "__main__":
    main()
