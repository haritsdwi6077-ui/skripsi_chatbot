"""Ingestion: baca PDF/TXT di data_source/, chunking, embedding, unggah ke Qdrant Cloud.

Dijalankan SECARA LOKAL (bukan di Vercel):

    python scripts/ingest.py --dry-run      # hitung halaman & chunk saja, tanpa API
    python scripts/ingest.py --recreate     # bangun ulang collection
    python scripts/ingest.py                # lanjutkan / tambah dokumen baru

Yang membuat proses lama BUKAN chunking (hanya hitungan milidetik), melainkan
embedding: setiap chunk harus dikirim ke API Jina. Karena itu skrip ini:

1. Mengirim embedding secara batch (banyak chunk per permintaan HTTP).
2. Menyimpan setiap embedding ke cache lokal (.cache/), sehingga bila proses
   terhenti (kuota habis, internet putus) atau collection dibuat ulang,
   chunk yang sudah di-embedding TIDAK dikirim ulang ke API.
3. Memakai ID chunk yang tetap (deterministik), sehingga menjalankan ulang
   tidak menggandakan data di Qdrant.
4. Membaca subfolder (misalnya data_source/Blas/...).
5. Melaporkan PDF yang tidak punya teks (hasil scan) agar bisa di-OCR atau dibuang.
6. Membaca PDF secara paralel (semua inti CPU) dengan PyMuPDF bila terpasang
   (jauh lebih cepat daripada pypdf), lalu menyimpan teks hasil ekstraksi ke
   .cache/teks/ sehingga PDF yang tidak berubah tidak dibaca ulang.
7. Bisa mengirim beberapa batch embedding sekaligus (--paralel).
8. Menyaring bagian tanaman lain. Dokumen multikomoditas (padi disebut minimal
   20 kali, dan jagung + kedelai minimal 25% dari seluruh sebutan komoditas)
   dipotong per bagian (judul Lampiran/Tabel/Gambar). Bagian yang lebih banyak
   menyebut jagung atau kedelai, termasuk istilah khasnya seperti tongkol, bulai,
   atau polong, daripada padi tidak dimasukkan. Matikan dengan --tanpa-saring.
9. Melewati halaman daftar isi/tabel/gambar (minimal 5 deret titik penghubung),
   karena hanya berisi judul dan nomor halaman, bukan isi.

Contoh cepat:
    pip install pymupdf
    python scripts/ingest.py --recreate --batch 100 --paralel 3 --jeda 5
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

logging.getLogger("pypdf").setLevel(logging.ERROR)  # sembunyikan peringatan font pypdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from rag_core import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_source"
CACHE_DIR = ROOT / ".cache"
NAMESPACE = uuid.UUID("7d3f5f0e-6a51-4c3e-9a0b-2b7f3c1d9e11")

# Penyaringan halaman tanaman lain pada dokumen multikomoditas.
# Selain nama tanaman, istilah yang hanya dipakai untuk tanaman itu ikut dihitung,
# karena tabel dan lampiran sering menyebut "P tongkol" atau "bulai" tanpa kata
# "jagung" di setiap halaman.
ISTILAH_KOMODITAS = {
    "padi": ("padi", "rice", "paddy"),
    "jagung": ("jagung", "tongkol", "bulai", "ostrinia", "helicoverpa", "maize", "corn"),
    "kedelai": ("kedelai", "polong", "riptortus", "etiella", "soybean"),
}
KOMODITAS_LAIN = ("jagung", "kedelai")
AMBANG_MULTIKOMODITAS = 0.25   # porsi sebutan jagung + kedelai dalam satu dokumen
MIN_SEBUTAN_PADI = 20          # dokumen harus membahas padi secara substansial
MIN_SEBUTAN_BAGIAN = 1         # bagian dibuang bila jagung/kedelai >= 1 dan > padi
# Satu halaman bisa memuat dua tabel (misal lampiran padi lalu lampiran jagung),
# jadi halaman dipotong di setiap judul Lampiran/Tabel/Gambar bernomor.
POLA_TITIK = re.compile(r"(?:\.\s?){6,}")  # deret titik penghubung di daftar isi


def halaman_daftar_isi(teks: str) -> bool:
    """True untuk halaman daftar isi/tabel/gambar (banyak deret titik)."""
    return len(POLA_TITIK.findall(teks)) >= 5


POLA_BAGIAN = re.compile(r"(?=\b(?:Lampiran|Tabel|Gambar)\s+\d+)")


def hitung_komoditas(teks: str) -> dict[str, int]:
    """Menghitung sebutan tiap komoditas beserta istilah khasnya (kata utuh)."""
    t = teks.lower()
    return {
        k: sum(len(re.findall(rf"\b{istilah}\b", t)) for istilah in daftar)
        for k, daftar in ISTILAH_KOMODITAS.items()
    }


def dokumen_multikomoditas(pages: list[list]) -> bool:
    """True bila dokumen membahas padi DAN jagung/kedelai secara substansial."""
    padi = lain = 0
    for _, teks in pages:
        c = hitung_komoditas(teks)
        padi += c["padi"]
        lain += sum(c[k] for k in KOMODITAS_LAIN)
    return padi >= MIN_SEBUTAN_PADI and lain / (padi + lain) >= AMBANG_MULTIKOMODITAS


def saring_tanaman_lain(pages: list[list]) -> tuple[list[list], int]:
    """Membuang bagian halaman yang membahas jagung/kedelai.

    Hanya berlaku untuk dokumen multikomoditas; dokumen lain dikembalikan utuh.
    Mengembalikan (halaman_baru, jumlah_bagian_dibuang).
    """
    if not dokumen_multikomoditas(pages):
        return pages, 0
    baru, dibuang = [], 0
    for no, teks in pages:
        simpan = []
        for bagian in POLA_BAGIAN.split(teks):
            if not bagian.strip():
                continue
            c = hitung_komoditas(bagian)
            lain = max(c[k] for k in KOMODITAS_LAIN)
            if lain >= MIN_SEBUTAN_BAGIAN and lain > c["padi"]:
                dibuang += 1
            else:
                simpan.append(bagian.strip())
        baru.append([no, "\n".join(simpan)])
    return baru, dibuang


def bersihkan(teks: str) -> str:
    """Membuang spasi berlebih dan pemenggalan kata di akhir baris."""
    teks = teks.replace("\r", "\n")
    teks = re.sub(r"-\n(?=\w)", "", teks)
    teks = re.sub(r"[ \t]+", " ", teks)
    teks = re.sub(r"\n{3,}", "\n\n", teks)
    return teks.strip()


def baca_pdf(path_str: str) -> tuple[list[list], str | None]:
    """Membaca satu PDF -> ([[nomor_halaman, teks], ...], nama_galat atau None).

    Memakai PyMuPDF bila terpasang (cepat, juga bisa membuka PDF terenkripsi AES),
    selain itu memakai pypdf. Fungsi ini dijalankan di proses terpisah.
    """
    try:
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf  # nama lama PyMuPDF
            except ImportError:
                pymupdf = None

        if pymupdf is not None:
            with pymupdf.open(path_str) as doc:
                return [[i, bersihkan(page.get_text())] for i, page in enumerate(doc, 1)], None

        from pypdf import PdfReader
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(path_str)
        return [[i, bersihkan(page.extract_text() or "")]
                for i, page in enumerate(reader.pages, 1)], None
    except Exception as exc:  # noqa: BLE001
        return [], exc.__class__.__name__


def _kunci_teks(path: Path) -> str:
    """Kunci cache teks: berubah bila isi file berubah (ukuran / waktu ubah)."""
    st = path.stat()
    mesin = "pymupdf" if _ada_pymupdf() else "pypdf"
    return hashlib.sha1(
        f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{mesin}".encode()
    ).hexdigest()


def _ada_pymupdf() -> bool:
    try:
        import pymupdf  # noqa: F401
        return True
    except ImportError:
        try:
            import fitz  # noqa: F401
            return True
        except ImportError:
            return False


def baca_dokumen(data_dir: Path | None = None, workers: int | None = None,
                 saring: bool = True) -> tuple[list[dict], list[str], dict[str, int]]:
    """Mengembalikan (halaman, pdf_tanpa_teks, halaman_disaring_per_dokumen)."""
    data_dir = data_dir or DATA_DIR
    cache_teks = CACHE_DIR / "teks"
    cache_teks.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]

    # 1) Ambil dari cache teks; sisanya dibaca paralel
    hasil_pdf: dict[Path, tuple[list[list], str | None]] = {}
    perlu_baca = []
    for p in pdfs:
        f = cache_teks / f"{_kunci_teks(p)}.json"
        if f.exists():
            try:
                hasil_pdf[p] = (json.loads(f.read_text(encoding="utf-8")), None)
                continue
            except (json.JSONDecodeError, OSError):
                pass
        perlu_baca.append(p)

    if perlu_baca:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        print(f"Membaca {len(perlu_baca)} PDF dengan {workers} proses "
              f"({'PyMuPDF' if _ada_pymupdf() else 'pypdf'}); "
              f"{len(pdfs) - len(perlu_baca)} PDF diambil dari cache.")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for p, (pages, galat) in zip(perlu_baca, ex.map(baca_pdf, map(str, perlu_baca))):
                hasil_pdf[p] = (pages, galat)
                if galat is None:
                    (cache_teks / f"{_kunci_teks(p)}.json").write_text(
                        json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    # 2) Susun halaman sesuai urutan file
    halaman, tanpa_teks, disaring = [], [], {}
    daftar_isi = 0
    for path in files:
        sumber = path.relative_to(data_dir).as_posix()
        akhiran = path.suffix.lower()
        if akhiran == ".pdf":
            pages, galat = hasil_pdf[path]
            if galat:
                tanpa_teks.append(f"{sumber} (gagal dibaca: {galat})")
                continue
            ada_teks = any(len(teks) > 50 for _, teks in pages)
            if saring:
                pages, n_buang = saring_tanaman_lain(pages)
                if n_buang:
                    disaring[sumber] = n_buang
            for nomor, teks in pages:
                if halaman_daftar_isi(teks):
                    daftar_isi += 1
                    continue
                if len(teks) > 50:
                    halaman.append({"source": sumber, "page": nomor, "text": teks})
            if not ada_teks:
                tanpa_teks.append(sumber)
        elif akhiran in {".txt", ".md"}:
            teks = bersihkan(path.read_text(encoding="utf-8", errors="ignore"))
            if teks:
                halaman.append({"source": sumber, "page": None, "text": teks})
    if daftar_isi:
        disaring["(halaman daftar isi, semua dokumen)"] = daftar_isi
    return halaman, tanpa_teks, disaring


def potong(teks: str, ukuran: int, tumpang_tindih: int) -> list[str]:
    """Chunking berbasis karakter dengan overlap, dipotong pada batas spasi."""
    potongan, mulai = [], 0
    while mulai < len(teks):
        akhir = min(mulai + ukuran, len(teks))
        if akhir < len(teks):
            spasi = teks.rfind(" ", mulai + int(ukuran * 0.6), akhir)
            if spasi != -1:
                akhir = spasi
        bagian = teks[mulai:akhir].strip()
        if len(bagian) > 40:
            potongan.append(bagian)
        if akhir >= len(teks):
            break
        mulai = max(akhir - tumpang_tindih, mulai + 1)
    return potongan


def buat_chunks(halaman: list[dict]) -> list[dict]:
    chunks = []
    for h in halaman:
        for urutan, bagian in enumerate(potong(h["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            kunci = f"{h['source']}|{h['page']}|{urutan}|{bagian}"
            chunks.append({
                "id": str(uuid.uuid5(NAMESPACE, kunci)),
                "hash": hashlib.sha1(bagian.encode("utf-8")).hexdigest(),
                "source": h["source"], "page": h["page"], "text": bagian,
                # topik = subfolder pertama di data_source (Blas, HDB, ...), "-" bila tidak ada
                "topik": h["source"].split("/", 1)[0] if "/" in h["source"] else "-",
            })
    return chunks


class CacheEmbedding:
    """Cache embedding per teks, disimpan sebagai JSON Lines (satu baris per chunk)."""

    def __init__(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        nama = f"embeddings_{config.EMBEDDING_MODEL}_{config.EMBEDDING_DIM}.jsonl"
        self.path = CACHE_DIR / nama
        self.data: dict[str, list[float]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for baris in f:
                    try:
                        rec = json.loads(baris)
                        self.data[rec["h"]] = rec["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # baris terakhir bisa terpotong bila proses dihentikan

    def simpan(self, hasil: dict[str, list[float]]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for h, v in hasil.items():
                f.write(json.dumps({"h": h, "v": [round(x, 7) for x in v]}) + "\n")
        self.data.update(hasil)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="hapus lalu buat ulang collection (cache embedding tetap dipakai)")
    parser.add_argument("--dry-run", action="store_true",
                        help="hanya baca PDF dan hitung chunk, tanpa memanggil API")
    parser.add_argument("--batch", type=int, default=50,
                        help="jumlah chunk per permintaan embedding (default 50)")
    parser.add_argument("--jeda", type=float, default=1.0,
                        help="jeda detik antar-gelombang permintaan embedding (atur sesuai kuota)")
    parser.add_argument("--paralel", type=int, default=1,
                        help="jumlah permintaan embedding yang dikirim bersamaan (default 1)")
    parser.add_argument("--tanpa-saring", action="store_true",
                        help="jangan saring bagian jagung/kedelai pada dokumen multikomoditas")
    parser.add_argument("--workers", type=int, default=None,
                        help="jumlah proses pembaca PDF (default: jumlah inti CPU - 1)")
    args = parser.parse_args()

    t0 = time.time()
    halaman, tanpa_teks, disaring = baca_dokumen(workers=args.workers,
                                                 saring=not args.tanpa_saring)
    if not halaman:
        raise SystemExit(f"Tidak ada dokumen berteks di {DATA_DIR}")
    chunks = buat_chunks(halaman)
    jumlah_dok = len({c["source"] for c in chunks})
    print(f"{jumlah_dok} dokumen, {len(halaman)} halaman -> {len(chunks)} chunk "
          f"(baca + chunking {time.time() - t0:.1f} detik)")
    n_daftar_isi = disaring.pop("(halaman daftar isi, semua dokumen)", 0)
    if disaring:
        print(f"\nDisaring {sum(disaring.values())} bagian halaman jagung/kedelai dari "
              f"{len(disaring)} dokumen multikomoditas:")
        for s, n in disaring.items():
            print(f"   - {s}: {n} bagian")
    if n_daftar_isi:
        print(f"Dilewati {n_daftar_isi} halaman daftar isi/tabel/gambar.")
    if tanpa_teks:
        print(f"\nPERINGATAN: {len(tanpa_teks)} PDF tidak menghasilkan teks "
              f"(kemungkinan hasil scan, perlu OCR atau dibuang):")
        for s in tanpa_teks:
            print("   -", s)

    cache = CacheEmbedding()
    belum = [c for c in chunks if c["hash"] not in cache.data]
    unik_belum = list({c["hash"]: c for c in belum}.values())
    n_req = -(-len(unik_belum) // args.batch) if unik_belum else 0
    print(f"\nSudah ada di cache: {len(chunks) - len(belum)} chunk | "
          f"perlu embedding: {len(unik_belum)} chunk unik "
          f"(~{n_req} permintaan batch @ {args.batch})")

    if args.dry_run:
        print("\nDry run selesai. Tidak ada API yang dipanggil.")
        return

    from qdrant_client import models  # noqa: PLC0415

    from database import vector_db  # noqa: PLC0415
    from rag_core import embeddings  # noqa: PLC0415

    # 1) Embedding yang belum ada di cache. Beberapa batch dikirim bersamaan
    #    (--paralel); hasil tiap batch langsung disimpan ke cache.
    kelompok = [unik_belum[i:i + args.batch] for i in range(0, len(unik_belum), args.batch)]
    selesai, t_embed = 0, time.time()

    def _embed(k: list[dict]) -> list[list[float]]:
        return embeddings.embed_documents([c["text"] for c in k])

    with ThreadPoolExecutor(max_workers=max(1, args.paralel)) as ex:
        for w in range(0, len(kelompok), max(1, args.paralel)):
            gelombang = kelompok[w:w + max(1, args.paralel)]
            futures = [ex.submit(_embed, k) for k in gelombang]
            galat = None
            for k, fut in zip(gelombang, futures):
                try:
                    vektor = fut.result()
                except Exception as exc:  # noqa: BLE001
                    galat = galat or exc
                    continue
                cache.simpan({c["hash"]: v for c, v in zip(k, vektor)})
                selesai += len(k)
            if galat:
                raise galat  # batch yang berhasil sudah tersimpan di cache
            laju = selesai / max(time.time() - t_embed, 1e-6)
            sisa = (len(unik_belum) - selesai) / laju if laju else 0
            print(f"  embedding {selesai}/{len(unik_belum)} "
                  f"(~{laju * 60:.0f} chunk/menit, sisa ~{sisa / 60:.1f} menit)")
            if selesai < len(unik_belum) and args.jeda:
                time.sleep(args.jeda)

    # 2) Unggah ke Qdrant (ID tetap -> menjalankan ulang tidak menggandakan data)
    vector_db.ensure_collection(recreate=args.recreate)
    for i in range(0, len(chunks), 128):
        kelompok = chunks[i:i + 128]
        vector_db.upsert([
            models.PointStruct(
                id=c["id"],
                vector=cache.data[c["hash"]],
                payload={"source": c["source"], "page": c["page"], "text": c["text"],
                         "topik": c["topik"]},
            )
            for c in kelompok
        ])
        print(f"  unggah {min(i + 128, len(chunks))}/{len(chunks)}")

    print(f"\nSelesai dalam {(time.time() - t0) / 60:.1f} menit. "
          f"Jumlah point dalam collection: {vector_db.count()}")


if __name__ == "__main__":
    main()
