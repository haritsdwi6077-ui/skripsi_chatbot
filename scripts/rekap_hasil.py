"""Rekap hasil evaluasi untuk tabel BAB IV (Tabel 4.18 sampai Tabel 4.22).

Jalankan setelah kedua mode dievaluasi dengan testset yang sama:

    python scripts/evaluate.py --mode rag --out hasil_rag.csv
    python scripts/evaluate.py --mode llm --out hasil_llm.csv
    python scripts/rekap_hasil.py --rag hasil_rag.csv --llm hasil_llm.csv

Keluaran ditampilkan di layar dan disimpan ke rekap_hasil.md. Skor yang kosong
(NaN, biasanya karena LLM penilai gagal mengurai jawaban) TIDAK dihitung sebagai
nol. Jumlah skor yang valid dilaporkan di kolom n agar terlihat jelas.

Uji sensitivitas: keluarkan soal yang penilaiannya diragukan (hasil verifikasi
manual), lalu uji Wilcoxon diulang pada soal yang tersisa:
    python scripts/rekap_hasil.py --rag hasil\\glm_rag_v2.csv --llm hasil\\glm_llm.csv --keluarkan 1,5,13

Butuh: pandas, scipy  (pip install scipy)
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import norm, wilcoxon

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
    """Median dan jumlah skor nol ikut dilaporkan, karena sebaran skor RAGAS
    (terutama factual correctness) miring dan banyak bernilai 0."""
    baris = []
    for k in kolom:
        if k not in df.columns:
            continue
        s = pd.to_numeric(df[k], errors="coerce").dropna()
        baris.append([METRIK_RAG[k], str(len(s)), angka(s.mean()), angka(s.median()),
                      angka(s.std(ddof=1)), angka(s.min()), angka(s.max()), str(int((s == 0).sum()))])
    return baris


def kategori_efek(r: float) -> str:
    """Batas Cohen untuk r: 0,1 kecil, 0,3 sedang, 0,5 besar."""
    if pd.isna(r):
        return "-"
    if r >= 0.5:
        return "besar"
    if r >= 0.3:
        return "sedang"
    if r >= 0.1:
        return "kecil"
    return "sangat kecil"


def uji_wilcoxon(gabung: pd.DataFrame, alpha: float) -> list[list[str]]:
    """Uji peringkat bertanda Wilcoxon berpasangan per pertanyaan, dengan ukuran
    efek r = Z / akar(n), n = jumlah pasangan yang skornya tidak sama."""
    baris = []
    for k in METRIK_BERSAMA:
        pasangan = gabung[[f"{k}_rag", f"{k}_llm"]].apply(pd.to_numeric, errors="coerce").dropna()
        x, y = pasangan[f"{k}_rag"], pasangan[f"{k}_llm"]
        beda = x - y
        naik, turun, sama = int((beda > 0).sum()), int((beda < 0).sum()), int((beda == 0).sum())
        n_beda = naik + turun
        if n_beda >= 1:
            uji = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            stat, nilai_p = uji.statistic, uji.pvalue
            z = norm.isf(nilai_p / 2) if nilai_p > 0 else float("inf")
            r = z / math.sqrt(n_beda)
            keputusan = "H0 ditolak" if nilai_p < alpha else "H0 diterima"
        else:
            stat, nilai_p, r, keputusan = float("nan"), float("nan"), float("nan"), "tidak dapat diuji"
        baris.append([METRIK_RAG[k], str(len(pasangan)), angka(x.mean()), angka(y.mean()),
                      angka(x.mean() - y.mean()), angka(stat, 1), angka(nilai_p),
                      angka(r, 3), kategori_efek(r), keputusan, f"{naik}/{turun}/{sama}"])
    return baris


HEAD_WILCOXON = ["Metrik", "n pasangan", "Dengan RAG", "Tanpa RAG", "Selisih", "Statistik T",
                 "Nilai p", "Ukuran efek r", "Kategori efek", "Keputusan",
                 "RAG lebih tinggi/lebih rendah/sama"]


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
    p.add_argument("--keluarkan", default=None,
                   help="nomor soal (mulai 1, urutan testset) yang dikeluarkan pada uji "
                        "sensitivitas, mis. 1,5,13")
    a = p.parse_args()

    rag = seragamkan(pd.read_csv(a.rag))
    llm = seragamkan(pd.read_csv(a.llm)) if Path(a.llm).exists() else None
    keluaran = [f"# Rekap hasil evaluasi\n\nBerkas: {a.rag}" + (f", {a.llm}" if llm is not None else "")]
    keluaran.append(f"\nJumlah pertanyaan: mode RAG = {len(rag)}"
                    + (f", mode tanpa RAG = {len(llm)}" if llm is not None else ""))

    head = ["Metrik", "n", "Rata-rata", "Median", "Simpangan baku", "Minimum", "Maksimum",
            "Jumlah skor 0"]
    keluaran.append(tabel_md("Tabel 4.18 Hasil evaluasi RAGAS mode LLM dengan RAG", head,
                             statistik(rag, list(METRIK_RAG))))
    if llm is not None:
        keluaran.append(tabel_md("Tabel 4.19 Hasil evaluasi RAGAS mode LLM tanpa RAG", head,
                                 statistik(llm, METRIK_BERSAMA)))

        # Uji peringkat bertanda Wilcoxon, berpasangan per pertanyaan
        gabung = rag.merge(llm, on="user_input", suffixes=("_rag", "_llm"))
        keluaran.append(tabel_md("Tabel 4.20 Perbandingan skor dan hasil uji Wilcoxon",
                                 HEAD_WILCOXON, uji_wilcoxon(gabung, a.alpha)))

        if a.keluarkan:
            nomor = sorted({int(x) for x in a.keluarkan.split(",") if x.strip()})
            soal = {rag.at[n - 1, "user_input"] for n in nomor if 1 <= n <= len(rag)}
            sisa = gabung[~gabung["user_input"].isin(soal)]
            keluaran.append(tabel_md(
                f"Uji sensitivitas: {len(soal)} soal dikeluarkan (no. "
                f"{', '.join(map(str, nomor))}), {len(sisa)} soal tersisa",
                HEAD_WILCOXON, uji_wilcoxon(sisa, a.alpha)))

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
