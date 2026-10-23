"""Membandingkan beberapa pengaturan retrieval (TOP_K dan MMR_LAMBDA) tanpa LLM penilai.

Ukuran yang dipakai adalah hit rate (Gao dkk., 2023): apakah potongan dari halaman
sumber jawaban rujukan ikut terpilih sebagai konteks. Setiap butir test set sudah
mencatat dokumen sumber ("sumber") dan nomor halamannya ("halaman"), jadi ukuran
ini bisa dihitung langsung dari hasil retrieval, tanpa memanggil LLM.

Dijalankan SECARA LOKAL (butuh JINA_API_KEY dan QDRANT_* di .env):
    python scripts/bandingkan_retrieval.py
    python scripts/bandingkan_retrieval.py --konfigurasi 4:0.6,6:0.6,4:0.75,6:0.75
    python scripts/bandingkan_retrieval.py --nomor 6,8,10,24,27,29,34,35

Embedding setiap pertanyaan hanya dihitung sekali dan hasil pencarian Qdrant
disimpan sementara, jadi menambah konfigurasi tidak menambah pemakaian kuota Jina.
Hasil per pertanyaan disimpan ke hasil/banding_retrieval.csv.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd  # noqa: E402

from database import vector_db  # noqa: E402
from rag_core import config, embeddings, engine  # noqa: E402

# --- cache: satu embedding per pertanyaan, satu pencarian per kombinasi argumen ---
_cache_embed: dict[str, list[float]] = {}
_cache_cari: dict[tuple, list[dict]] = {}
_embed_asli = embeddings.embed_query
_cari_asli = vector_db.search


def _embed(teks: str) -> list[float]:
    if teks not in _cache_embed:
        _cache_embed[teks] = _embed_asli(teks)
    return _cache_embed[teks]


def _cari(vector, top_k, with_vectors=False, topik=None):
    kunci = (tuple(round(v, 6) for v in vector[:16]), top_k, with_vectors, topik)
    if kunci not in _cache_cari:
        _cache_cari[kunci] = _cari_asli(vector, top_k, with_vectors=with_vectors, topik=topik)
    return [dict(x) for x in _cache_cari[kunci]]


embeddings.embed_query = _embed
engine.embeddings.embed_query = _embed
vector_db.search = _cari
engine.vector_db.search = _cari


def nilai_satu(item: dict, toleransi: int) -> dict:
    """Retrieval untuk satu pertanyaan dengan pengaturan config saat ini."""
    k, _, dipilih = engine.ambil_konteks(item["question"])
    sumber, hlm = item.get("sumber"), item.get("halaman")
    hit_dok = any(c["source"] == sumber for c in dipilih)
    peringkat = None
    for i, c in enumerate(dipilih, 1):
        if (c["source"] == sumber and c.get("page") is not None and hlm is not None
                and abs(int(c["page"]) - int(hlm)) <= toleransi):
            peringkat = i
            break
    return {"k": k, "hit_dokumen": hit_dok, "hit_halaman": peringkat is not None,
            "peringkat_hit": peringkat,
            "halaman_terpilih": "; ".join(f"{Path(c['source']).name[:25]} hlm.{c.get('page')}"
                                          for c in dipilih)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data_source/testset.json")
    ap.add_argument("--konfigurasi", default="4:0.6,6:0.6,4:0.75,6:0.75",
                    help="daftar TOP_K:MMR_LAMBDA dipisah koma; yang pertama dianggap rancangan awal")
    ap.add_argument("--toleransi-halaman", type=int, default=1,
                    help="selisih nomor halaman yang masih dihitung tepat (default 1)")
    ap.add_argument("--nomor", default=None, help="hanya nomor soal tertentu, mis. 6,8,10 (mulai 1)")
    ap.add_argument("--out", default="hasil/banding_retrieval.csv")
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    nomor = ([int(n) for n in args.nomor.split(",") if n.strip()] if args.nomor
             else list(range(1, len(testset) + 1)))
    konfigurasi = []
    for bagian in args.konfigurasi.split(","):
        k, lam = bagian.split(":")
        konfigurasi.append((int(k), float(lam)))

    baris = []
    for top_k, lam in konfigurasi:
        config.TOP_K, config.MMR_LAMBDA = top_k, lam
        print(f"TOP_K={top_k}, MMR_LAMBDA={lam} ...")
        for n in nomor:
            item = testset[n - 1]
            hasil = nilai_satu(item, args.toleransi_halaman)
            baris.append({"no": n, "jenis": item.get("jenis", "-"), "pertanyaan": item["question"],
                          "TOP_K": top_k, "MMR_LAMBDA": lam, **hasil})

    df = pd.DataFrame(baris)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    def label(r):
        return f"TOP_K={r['TOP_K']}, λ={str(r['MMR_LAMBDA']).replace('.', ',')}"

    df["konfigurasi"] = df.apply(label, axis=1)
    urutan = list(dict.fromkeys(df["konfigurasi"]))

    print(f"\nHit rate halaman sumber (toleransi ±{args.toleransi_halaman} halaman), "
          f"{len(nomor)} pertanyaan:\n")
    print("| Konfigurasi | Hit dokumen | Hit halaman | Peringkat rata-rata |")
    print("|---|---|---|---|")
    for kf in urutan:
        d = df[df.konfigurasi == kf]
        rata = d["peringkat_hit"].dropna().mean()
        print(f"| {kf} | {d.hit_dokumen.mean():.0%} | {d.hit_halaman.mean():.0%} | "
              f"{'-' if pd.isna(rata) else f'{rata:.1f}'.replace('.', ',')} |")

    print("\nHit halaman per jenis pertanyaan:\n")
    tabel = df.pivot_table(index="jenis", columns="konfigurasi", values="hit_halaman", aggfunc="mean")
    print("| Jenis | " + " | ".join(urutan) + " |")
    print("|---|" + "---|" * len(urutan))
    for jenis, r in tabel[urutan].iterrows():
        print(f"| {jenis} | " + " | ".join(f"{v:.0%}" for v in r) + " |")

    awal = df[df.konfigurasi == urutan[0]].set_index("no")["hit_halaman"]
    for kf in urutan[1:]:
        lain = df[df.konfigurasi == kf].set_index("no")["hit_halaman"]
        naik = [n for n in awal.index if not awal[n] and lain[n]]
        turun = [n for n in awal.index if awal[n] and not lain[n]]
        print(f"\n{kf} dibanding {urutan[0]}: membaik pada soal {naik or '-'}, "
              f"memburuk pada soal {turun or '-'}")
    print(f"\nRincian per pertanyaan: {args.out}")


if __name__ == "__main__":
    main()
