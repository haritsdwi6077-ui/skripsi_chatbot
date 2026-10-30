"""Mengolah jawaban kuesioner kepuasan pengguna Tanya Padi dengan skala Likert 1-5.

Langkah perhitungan mengikuti Subbab 2.13:
  1. Setiap jawaban diberi bobot 1 (sangat tidak setuju) sampai 5 (sangat setuju).
  2. Skor total butir      = jumlah (bobot x frekuensi)          (Persamaan 2.14)
     Rata-rata butir       = skor total / jumlah responden
     Persentase skor       = skor total / (5 x jumlah responden) x 100%
  3. Uji validitas         : korelasi Pearson butir dengan skor total seluruh butir
                             (--validitas total, bawaan) atau skor total aspeknya
                             (--validitas aspek),
                             valid bila r hitung > r tabel (alfa 0,05, df = n - 2).
                             Korelasi butir-total terkoreksi juga dilaporkan;
                             butir < 0,30 perlu ditinjau (Boateng dkk., 2018).
  4. Uji reliabilitas      : Cronbach's alpha per aspek (Persamaan 2.13), reliabel
                             bila alpha >= 0,70 (Taber, 2018).
  5. Skor aspek            : rata-rata skor butir yang valid di dalam aspek itu.
  6. Kategori              : interval (5 - 1) / 5 = 0,8 (Persamaan 2.15, Tabel 2.2).

Cara memakai:
  1. Buka Google Form > tab Jawaban > ikon Spreadsheet (Tautkan ke Spreadsheet).
  2. Di Google Sheets: File > Download > Comma Separated Values (.csv).
  3. Simpan berkasnya, misalnya di hasil/kuesioner.csv, lalu jalankan:
         python scripts/olah_kuesioner.py --csv hasil/kuesioner.csv

Keluaran:
  hasil/kuesioner_rekap.md           tabel siap salin ke Subbab 4.8
  hasil/kuesioner_per_butir.csv      rincian per butir (untuk lampiran)
  hasil/kuesioner_jawaban_terbuka.csv jawaban pertanyaan terbuka, siap diberi kode tema

Butuh: pandas, scipy (sudah ada di requirements-dev.txt).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Kode singkat per aspek (judul grid di Google Form)
KODE_ASPEK = {
    "Kegunaan": "KG",
    "Kemudahan Penggunaan": "KM",
    "Kualitas Jawaban": "KJ",
    "Keandalan dan Kepercayaan": "KK",
    "Percakapan dan Kecepatan": "PK",
    "Kepuasan Keseluruhan": "KP",
}
# Kuesioner EUCS 12 butir bernomor ("1. Pernyataan"), Doll & Torkzadeh (1988)
ASPEK_NOMOR = [
    (1, 4, "Isi", "C"),
    (5, 6, "Keakuratan", "A"),
    (7, 8, "Tampilan", "F"),
    (9, 10, "Kemudahan Penggunaan", "E"),
    (11, 12, "Ketepatan Waktu", "T"),
]
LABEL_BOBOT = {
    "sangat tidak setuju": 1, "tidak setuju": 2, "netral": 3, "ragu-ragu": 3,
    "cukup setuju": 3, "setuju": 4, "sangat setuju": 5,
}
KATA_PROFIL = ("usia", "jenis kelamin", "status", "pengalaman", "perangkat")
KATA_TERBUKA = ("hal apa", "pertanyaan tentang", "fitur atau informasi", "saran", "masukan")
KATEGORI = [(1.80, "Sangat tidak puas"), (2.60, "Tidak puas"), (3.40, "Cukup puas"),
            (4.20, "Puas"), (5.00, "Sangat puas")]


def angka(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{d}f}".replace(".", ",")


def persen(v, d=1):
    return "-" if v is None or np.isnan(v) else f"{v:.{d}f}%".replace(".", ",")


def kategori(rata):
    if np.isnan(rata):
        return "-"
    for batas, nama in KATEGORI:
        if round(rata, 2) <= batas:
            return nama
    return KATEGORI[-1][1]


def bobot(nilai):
    """'4 - Setuju' -> 4, 'Setuju' -> 4, 4 -> 4."""
    if pd.isna(nilai):
        return np.nan
    teks = str(nilai).strip()
    m = re.match(r"^\s*([1-5])\b", teks)
    if m:
        return int(m.group(1))
    return LABEL_BOBOT.get(teks.lower(), np.nan)


def r_tabel(n, alfa=0.05):
    df = n - 2
    if df < 1:
        return np.nan
    t = stats.t.ppf(1 - alfa / 2, df)
    return t / np.sqrt(df + t ** 2)


def cronbach(data: pd.DataFrame) -> float:
    data = data.dropna()
    k = data.shape[1]
    if k < 2 or len(data) < 2:
        return np.nan
    var_butir = data.var(axis=0, ddof=1).sum()
    var_total = data.sum(axis=1).var(ddof=1)
    return np.nan if var_total == 0 else k / (k - 1) * (1 - var_butir / var_total)


def tabel_md(header, baris):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in b) + " |" for b in baris]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV jawaban dari Google Sheets")
    ap.add_argument("--out", default="hasil/kuesioner_rekap.md")
    ap.add_argument("--alfa", type=float, default=0.05)
    ap.add_argument("--validitas", choices=["total", "aspek"], default="total",
                    help="pembanding korelasi butir: skor total seluruh butir atau skor total aspek")
    ap.add_argument("--sertakan-tidak-valid", action="store_true",
                    help="tetap hitung butir tidak valid dalam skor aspek")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    jumlah_awal = len(df)

    # Responden yang tidak bersedia tidak dihitung
    kol_setuju = next((c for c in df.columns if "bersedia" in c.lower()), None)
    if kol_setuju:
        df = df[df[kol_setuju].fillna("").str.lower().str.startswith("ya")]

    # Kolom grid Likert: "Aspek [pernyataan]"
    butir = []
    for c in df.columns:
        m = re.match(r"^(.+?)\s*\[(.+)\]\s*$", c)
        if m:
            butir.append((c, m.group(1).strip(), m.group(2).strip()))
    if not butir:
        # Format lain: kolom bernomor "1. Pernyataan", aspek ditentukan ASPEK_NOMOR
        for c in df.columns:
            m = re.match(r"^(\d+)\.\s+(.+)$", c)
            if not m:
                continue
            no = int(m.group(1))
            for awal, akhir, nama, kode in ASPEK_NOMOR:
                if awal <= no <= akhir:
                    KODE_ASPEK[nama] = kode
                    butir.append((c, nama, m.group(2).strip()))
    if not butir:
        raise SystemExit("BERHENTI: tidak ada kolom berformat 'Aspek [pernyataan]'. "
                         "Pastikan CSV diunduh dari Google Sheets jawaban form.")
    skor = pd.DataFrame({c: df[c].map(bobot) for c, _, _ in butir}, index=df.index)
    skor = skor.dropna(how="all")
    df = df.loc[skor.index]
    n = len(skor)
    if n < 3:
        raise SystemExit(f"BERHENTI: baru {n} responden yang mengisi pernyataan; minimal 3.")

    # Kode butir: KG1, KG2, ...
    hitung, info = {}, []
    for c, aspek, teks in butir:
        prefix = KODE_ASPEK.get(aspek) or "".join(w[0] for w in aspek.split()).upper()
        hitung[prefix] = hitung.get(prefix, 0) + 1
        info.append({"kolom": c, "aspek": aspek, "kode": f"{prefix}{hitung[prefix]}", "pernyataan": teks})
    info = pd.DataFrame(info)
    skor.columns = info["kode"]
    urutan_aspek = list(dict.fromkeys(info["aspek"]))
    rt = r_tabel(n, a.alfa)

    # --- per butir: frekuensi, skor total, rata-rata, validitas ---
    rinci = []
    for aspek in urutan_aspek:
        kode = info.loc[info.aspek == aspek, "kode"].tolist()
        acuan = skor.columns.tolist() if a.validitas == "total" else kode
        total_aspek = skor[acuan].sum(axis=1, min_count=1)
        for k in kode:
            s = skor[k]
            f = [int((s == j).sum()) for j in range(1, 6)]
            nk = int(s.notna().sum())
            total = int(sum(j * fj for j, fj in zip(range(1, 6), f)))
            rata = total / nk if nk else np.nan
            ok = s.notna() & total_aspek.notna()
            r = s[ok].corr(total_aspek[ok]) if ok.sum() > 2 else np.nan
            rk = s[ok].corr(total_aspek[ok] - s[ok]) if ok.sum() > 2 and len(acuan) > 1 else np.nan
            rinci.append({
                "aspek": aspek, "kode": k,
                "pernyataan": info.loc[info.kode == k, "pernyataan"].iloc[0],
                "STS": f[0], "TS": f[1], "N": f[2], "S": f[3], "SS": f[4], "n": nk,
                "skor_total": total, "rata_rata": rata,
                "persen": total / (5 * nk) * 100 if nk else np.nan,
                "kategori": kategori(rata), "r_hitung": r, "r_tabel": rt,
                "valid": bool(r > rt) if not np.isnan(r) else False,
                "r_terkoreksi": rk,
            })
    rinci = pd.DataFrame(rinci)

    # --- per aspek: reliabilitas dan skor ---
    per_aspek = []
    for aspek in urutan_aspek:
        d = rinci[rinci.aspek == aspek]
        dipakai = d if a.sertakan_tidak_valid else d[d.valid]
        alpha = cronbach(skor[dipakai.kode.tolist()]) if len(dipakai) > 1 else np.nan
        rata = dipakai.rata_rata.mean() if len(dipakai) else np.nan
        per_aspek.append({"aspek": aspek, "butir": len(d), "valid": int(d.valid.sum()),
                          "alpha": alpha, "rata_rata": rata,
                          "persen": dipakai.skor_total.sum() / (5 * dipakai.n.sum()) * 100 if len(dipakai) else np.nan,
                          "kategori": kategori(rata)})
    per_aspek = pd.DataFrame(per_aspek)
    semua = rinci if a.sertakan_tidak_valid else rinci[rinci.valid]
    rata_all = semua.rata_rata.mean()
    alpha_all = cronbach(skor[semua.kode.tolist()]) if len(semua) > 1 else np.nan

    # --- profil responden ---
    profil = []
    for c in df.columns:
        cl = c.lower()
        if c in skor.columns or "[" in c or not any(cl.startswith(k) for k in KATA_PROFIL):
            continue
        hit = df[c].fillna("(kosong)").value_counts()
        for nilai, jml in hit.items():
            profil.append([c, nilai, jml, persen(jml / n * 100)])

    # --- jawaban terbuka ---
    terbuka = [c for c in df.columns if any(cl in c.lower() for cl in KATA_TERBUKA) and "[" not in c]
    if terbuka:
        jt = df[terbuka].copy()
        jt.insert(0, "responden", [f"R{i:02d}" for i in range(1, len(jt) + 1)])
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        jt.to_csv(Path(a.out).with_name("kuesioner_jawaban_terbuka.csv"), index=False, encoding="utf-8-sig")

    # --- tulis rekap ---
    L = ["# Rekap kuesioner kepuasan pengguna Tanya Padi", "",
         f"Berkas: {a.csv}", "",
         f"Jumlah baris jawaban: {jumlah_awal}; responden yang dihitung: {n} "
         f"(bersedia dan mengisi pernyataan).", ""]
    if profil:
        L += ["### Profil responden", "", tabel_md(["Pertanyaan", "Jawaban", "Jumlah", "Persentase"], profil), ""]
    L += [f"### Uji validitas (r tabel = {angka(rt, 3)}, n = {n}, alfa = {angka(a.alfa)}, "
          f"pembanding: skor total {'seluruh butir' if a.validitas == 'total' else 'aspek'})", "",
          tabel_md(["Kode", "r hitung", "r tabel", "Keterangan", "r butir-total terkoreksi"],
                   [[r.kode, angka(r.r_hitung, 3), angka(r.r_tabel, 3),
                     "Valid" if r.valid else "Tidak valid", angka(r.r_terkoreksi, 3)]
                    for r in rinci.itertuples()]), ""]
    L += ["### Uji reliabilitas (Cronbach's alpha)", "",
          tabel_md(["Aspek", "Jumlah butir valid", "Cronbach's alpha", "Keterangan"],
                   [[p.aspek, p.valid, angka(p.alpha, 3),
                     "-" if np.isnan(p.alpha) else ("Reliabel" if p.alpha >= 0.70 else "Kurang reliabel")]
                    for p in per_aspek.itertuples()]
                   + [["Seluruh butir valid", len(semua), angka(alpha_all, 3),
                       "-" if np.isnan(alpha_all) else ("Reliabel" if alpha_all >= 0.70 else "Kurang reliabel")]]), ""]
    L += ["### Rekap jawaban per butir", "",
          "STS = 1, TS = 2, N = 3, S = 4, SS = 5. Skor total = jumlah (bobot x frekuensi).", "",
          tabel_md(["Kode", "STS", "TS", "N", "S", "SS", "Skor total", "Rata-rata", "Persentase", "Kategori"],
                   [[r.kode, r.STS, r.TS, r.N, r.S, r.SS, r.skor_total, angka(r.rata_rata),
                     persen(r.persen), r.kategori] for r in rinci.itertuples()]), ""]
    L += ["### Skor per aspek", "",
          tabel_md(["Aspek", "Rata-rata", "Persentase", "Kategori"],
                   [[p.aspek, angka(p.rata_rata), persen(p.persen), p.kategori] for p in per_aspek.itertuples()]
                   + [["**Keseluruhan**", f"**{angka(rata_all)}**",
                       f"**{persen(semua.skor_total.sum() / (5 * semua.n.sum()) * 100)}**",
                       f"**{kategori(rata_all)}**"]]), ""]
    tinggi = per_aspek.loc[per_aspek.rata_rata.idxmax()]
    rendah = per_aspek.loc[per_aspek.rata_rata.idxmin()]
    L += ["### Ringkasan", "",
          f"- Aspek tertinggi: {tinggi.aspek} ({angka(tinggi.rata_rata)}, {tinggi.kategori.lower()}).",
          f"- Aspek terendah: {rendah.aspek} ({angka(rendah.rata_rata)}, {rendah.kategori.lower()}).",
          f"- Butir tertinggi: {rinci.loc[rinci.rata_rata.idxmax(), 'kode']} "
          f"({angka(rinci.rata_rata.max())}); butir terendah: {rinci.loc[rinci.rata_rata.idxmin(), 'kode']} "
          f"({angka(rinci.rata_rata.min())}).",
          f"- Butir tidak valid: {', '.join(rinci.loc[~rinci.valid, 'kode']) or 'tidak ada'}.", ""]
    L += ["### Daftar kode butir", "",
          tabel_md(["Kode", "Aspek", "Pernyataan"], [[r.kode, r.aspek, r.pernyataan] for r in rinci.itertuples()]), ""]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    rinci.to_csv(Path(a.out).with_name("kuesioner_per_butir.csv"), index=False, encoding="utf-8-sig")
    print("\n".join(L))
    print(f"\nDisimpan ke {a.out}, kuesioner_per_butir.csv"
          + (", kuesioner_jawaban_terbuka.csv" if terbuka else ""))


if __name__ == "__main__":
    main()
