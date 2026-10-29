"""Mesin RAG: retrieval dari Qdrant Cloud, penyusunan prompt, lalu generation.

Versi sebelumnya memakai rantai LangChain dengan Gemini. Versi ini memanggil
API secara langsung agar dependensinya ringan (penting untuk deploy di Vercel)
dan agar model LLM mudah diganti untuk kebutuhan perbandingan antar-LLM.
"""

import re
import time

from rag_core import config, embeddings, llm
from rag_core.pertanyaan import lengkapi_pertanyaan
from database import vector_db

# Aturan tampilan jawaban. Dipakai di KEDUA mode agar perbandingan tetap adil.
RANAH = (
    " Jika pertanyaan membahas tanaman selain padi, sampaikan dengan sopan bahwa "
    "asisten ini khusus untuk hama dan penyakit tanaman padi."
)

FORMAT_JAWABAN = (
    " Format jawaban: tulis paragraf pendek (2-3 kalimat) dan gunakan daftar "
    "berpoin bila ada beberapa butir. Beri baris kosong di antara paragraf dan "
    "sebelum daftar. Boleh memakai subjudul singkat yang diawali '### '. Jangan "
    "memakai tabel, tag HTML seperti <br>, atau garis pemisah."
)

SYSTEM_PROMPT = (
    "Anda adalah asisten penyuluh pertanian yang menjawab pertanyaan seputar "
    "hama dan penyakit tanaman padi. Jawablah HANYA berdasarkan konteks yang "
    "diberikan. Dokumen konteks dapat berbahasa Indonesia maupun Inggris, "
    "tetapi jawaban selalu ditulis dalam bahasa Indonesia yang ringkas dan "
    "mudah dipahami petani. Jika konteks tidak memuat informasi yang "
    "ditanyakan, katakan bahwa informasi tersebut tidak ditemukan pada dokumen "
    "yang diambil untuk pertanyaan ini, lalu sarankan pengguna bertanya lebih "
    "spesifik, misalnya menyebut nama hama atau penyakitnya. Jangan mengarang "
    "jawaban. Jika pertanyaan meminta daftar atau gambaran umum, sebutkan semua "
    "hama atau penyakit yang ada di konteks, lalu jelaskan bahwa daftar itu "
    "hanya mencakup dokumen yang diambil. Cantumkan nomor sumber, "
    "misalnya [1], pada bagian jawaban yang mengambil informasi dari konteks "
    "tersebut. Jangan menambahkan hama, penyakit, atau keterangan yang tidak "
    "ada di konteks. Jangan menebak kepanjangan singkatan yang tidak dijelaskan "
    "di konteks; tulis singkatannya saja. Bila sebagian konteks membahas tanaman lain seperti jagung "
    "atau kedelai, abaikan bagian itu dan jawab hanya untuk tanaman padi."
    + RANAH
    + FORMAT_JAWABAN
)

# Prompt pembanding (LLM saja, tanpa retrieval). Peran, bahasa, dan gaya jawaban
# dibuat sama dengan SYSTEM_PROMPT agar perbedaan hasil hanya berasal dari ada
# atau tidaknya konteks hasil retrieval.
SYSTEM_PROMPT_TANPA_RAG = (
    ""
)

# --- Percakapan berkelanjutan ---------------------------------------------
# Pertanyaan lanjutan ("cara pencegahannya gimana?") tidak menyebut nama hama
# atau penyakitnya, sehingga pencarian dokumen akan meleset. Sebelum retrieval,
# LLM menulis ulang pertanyaan itu menjadi pertanyaan mandiri berdasarkan
# riwayat percakapan (teknik "condense question" / query rewriting).
MAKS_RIWAYAT = 3              # jumlah giliran tanya-jawab terakhir yang dipakai
MAKS_JAWABAN_RIWAYAT = 800    # jawaban lama dipotong agar prompt tidak membengkak

SYSTEM_PROMPT_TULIS_ULANG = (
    "Tugas Anda menulis ulang pertanyaan lanjutan pengguna menjadi satu "
    "pertanyaan mandiri yang bisa dipahami tanpa membaca percakapan sebelumnya. "
    "Ganti kata rujukan seperti 'itu', 'tersebut', atau akhiran '-nya' dengan "
    "nama hama, penyakit, atau hal yang dimaksud dari percakapan. Jika "
    "pertanyaan sudah jelas berdiri sendiri, tulis kembali apa adanya. Jangan "
    "menjawab pertanyaan. Tulis hanya pertanyaannya dalam satu kalimat bahasa "
    "Indonesia, tanpa penjelasan tambahan."
)


def rapikan_riwayat(riwayat: list[dict] | None) -> list[dict]:
    """Ambil giliran terakhir yang valid, jawaban lama dipotong."""
    hasil = []
    for g in (riwayat or [])[-MAKS_RIWAYAT:]:
        q = str(g.get("question") or "").strip()
        a = str(g.get("answer") or "").strip()
        if q and a:
            if len(a) > MAKS_JAWABAN_RIWAYAT:
                a = a[:MAKS_JAWABAN_RIWAYAT].rsplit(" ", 1)[0] + " ..."
            hasil.append({"question": q, "answer": a})
    return hasil


def tulis_ulang_pertanyaan(question: str, riwayat: list[dict],
                           model: str | None = None) -> tuple[str, float]:
    """Mengubah pertanyaan lanjutan menjadi pertanyaan mandiri.

    Mengembalikan (pertanyaan_mandiri, detik). Tanpa riwayat, pertanyaan
    dikembalikan apa adanya. Bila LLM gagal atau hasilnya janggal, dipakai
    gabungan pertanyaan sebelumnya dan pertanyaan sekarang sebagai cadangan.
    """
    if not riwayat:
        return question, 0.0
    cadangan = f"{riwayat[-1]['question']} {question}"
    percakapan = "\n\n".join(
        f"Pengguna: {g['question']}\nAsisten: {g['answer']}" for g in riwayat
    )
    prompt = (f"Percakapan sebelumnya:\n{percakapan}\n\n"
              f"Pertanyaan lanjutan: {question}\n\nPertanyaan mandiri:")
    try:
        hasil = llm.generate_detail(
            prompt,
            system_prompt=SYSTEM_PROMPT_TULIS_ULANG,
            model=model,
            temperature=0,
            reasoning_effort="low" if config.LLM_REASONING_EFFORT else "",
        )
    except Exception:  # noqa: BLE001
        return cadangan, 0.0
    teks = (hasil["text"] or "").strip().splitlines()
    teks = teks[0].strip().strip('"\'') if teks else ""
    if teks.lower().startswith("pertanyaan mandiri:"):
        teks = teks.split(":", 1)[1].strip()
    if not teks or len(teks) > 400:
        return cadangan, hasil["latency_s"]
    return teks, hasil["latency_s"]


PROMPT_TEMPLATE = """Konteks:
{context}

Pertanyaan pengguna:
{question}

Jawaban:"""


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def topik_dari(source: str) -> str | None:
    """Topik = nama subfolder pertama di data_source, misalnya 'Blas/007_x.pdf' -> 'Blas'."""
    return source.split("/", 1)[0] if "/" in (source or "") else None


def pilih_beragam(kandidat: list[dict], top_k: int, lam: float | None = None,
                  maks_per_sumber: int | None = None,
                  jendela: float | None = None) -> list[dict]:
    """Memilih top_k potongan yang relevan sekaligus beragam.

    Dasarnya Maximal Marginal Relevance (MMR):
        skor = lam * kemiripan(pertanyaan, potongan)
               - (1 - lam) * kemiripan tertinggi(potongan, potongan terpilih)
    Vektor Jina sudah dinormalisasi, jadi kemiripan = hasil kali titik.

    Pemilihan berjalan dalam tiga tahap, makin longgar:
      1. Hanya kandidat yang skornya dalam `jendela` dari skor tertinggi, dan
         tiap topik (subfolder) serta tiap dokumen baru boleh 1 potongan.
         Pada pertanyaan umum ("penyakit padi apa saja?") skor antartopik
         berdekatan, jadi tahap ini memunculkan beberapa topik sekaligus.
      2. Semua kandidat, tiap dokumen maksimal `maks_per_sumber` potongan.
      3. Semua kandidat tanpa batas, bila masih kurang dari top_k.
    Kunci "vector" dibuang dari hasil; kunci "topik" ditambahkan.
    """
    lam = config.MMR_LAMBDA if lam is None else lam
    maks = config.MAKS_PER_SUMBER if maks_per_sumber is None else maks_per_sumber
    jendela = config.JENDELA_SKOR if jendela is None else jendela
    sisa = [dict(k, topik=topik_dari(k["source"])) for k in kandidat if k.get("vector")]
    if not sisa:
        return []
    tertinggi = max(k["score"] for k in sisa)
    terpilih: list[dict] = []
    per_sumber: dict[str, int] = {}
    per_topik: dict[str, int] = {}

    def boleh(k: dict, tahap: int) -> bool:
        if tahap == 1:
            if jendela <= 0 or k["score"] < tertinggi - jendela:
                return False
            if k["topik"] and per_topik.get(k["topik"], 0) >= 1:
                return False
            return per_sumber.get(k["source"], 0) < 1
        if tahap == 2:
            return per_sumber.get(k["source"], 0) < maks
        return True

    for tahap in (1, 2, 3):
        while sisa and len(terpilih) < top_k:
            terbaik, skor_terbaik = None, float("-inf")
            for k in sisa:
                if not boleh(k, tahap):
                    continue
                mirip = max((_dot(k["vector"], t["vector"]) for t in terpilih), default=0.0)
                skor = lam * k["score"] - (1 - lam) * mirip
                if skor > skor_terbaik:
                    terbaik, skor_terbaik = k, skor
            if terbaik is None:
                break
            terpilih.append(terbaik)
            sisa.remove(terbaik)
            per_sumber[terbaik["source"]] = per_sumber.get(terbaik["source"], 0) + 1
            if terbaik["topik"]:
                per_topik[terbaik["topik"]] = per_topik.get(terbaik["topik"], 0) + 1
    return [{k: v for k, v in t.items() if k != "vector"} for t in terpilih]


# Penanda pertanyaan bertipe daftar
POLA_DAFTAR = re.compile(
    r"\b(apa\s+saja|apa-apa\s+saja|sebutkan|jenis[\s-]+jenis|macam[\s-]+macam|"
    r"daftar|berbagai|what\s+are|list\s+of)\b",
    re.IGNORECASE,
)


def jumlah_konteks(question: str, top_k: int | None = None) -> int:
    """Jumlah potongan yang diambil: top_k bila diberikan, TOP_K_DAFTAR untuk
    pertanyaan bertipe daftar, selain itu TOP_K."""
    if top_k:
        return top_k
    if POLA_DAFTAR.search(question or ""):
        return max(config.TOP_K, config.TOP_K_DAFTAR)
    return config.TOP_K


def pilih_per_topik(kandidat: list[dict], top_k: int) -> list[dict]:
    """Untuk pertanyaan daftar: satu potongan terbaik dari setiap topik.

    Topik diurutkan dari skor potongan terbaiknya, lalu diambil top_k topik
    teratas. Bila jumlah topik kurang dari top_k, sisa tempat diisi dengan
    pilih_beragam() dari kandidat yang belum terpilih.
    """
    terbaik: dict[str, dict] = {}
    for k in kandidat:
        t = topik_dari(k["source"])
        if t and (t not in terbaik or k["score"] > terbaik[t]["score"]):
            terbaik[t] = k
    terpilih = sorted(terbaik.values(), key=lambda k: -k["score"])[:top_k]
    if len(terpilih) < top_k:
        sudah = {id(k) for k in terpilih}
        sisa = [k for k in kandidat if id(k) not in sudah]
        terpilih += [dict(k, vector=k.get("vector")) for k in pilih_beragam(
            [k for k in sisa if k.get("vector")], top_k - len(terpilih))]
    return [dict({x: y for x, y in k.items() if x != "vector"}, topik=topik_dari(k["source"]))
            for k in terpilih]


def ambil_konteks(question: str, top_k: int | None = None) -> tuple[int, list[dict], list[dict]]:
    """Retrieval lengkap. Mengembalikan (k, kandidat, terpilih).

    - Embedding pertanyaan, lalu ambil RETRIEVAL_KANDIDAT kandidat termirip.
    - Pertanyaan bertipe daftar dan collection punya topik: tambahkan 2 kandidat
      terbaik dari SETIAP topik (pencarian dengan filter topik), lalu ambil satu
      potongan per topik. Dengan begitu topik yang tidak masuk 40 besar, misalnya
      tungro, tetap bisa muncul.
    - Selain itu: pilih_beragam() (MMR + keragaman topik), atau top-k biasa bila
      MMR_AKTIF=0.
    Dipakai oleh retrieve() dan scripts/cek_retrieval.py agar hasilnya sama.
    """
    k = jumlah_konteks(question, top_k)
    vector = embeddings.embed_query(question)
    if not config.MMR_AKTIF:
        hasil = vector_db.search(vector, k)
        return k, hasil, hasil
    kandidat = vector_db.search(vector, max(config.RETRIEVAL_KANDIDAT, k), with_vectors=True)
    topik = vector_db.daftar_topik() if k > config.TOP_K else ()
    if topik:
        ada = {(c["source"], c["page"], c["text"]) for c in kandidat}
        for t in topik:
            for c in vector_db.search(vector, 2, with_vectors=True, topik=t):
                if (c["source"], c["page"], c["text"]) not in ada:
                    kandidat.append(c)
                    ada.add((c["source"], c["page"], c["text"]))
        kandidat.sort(key=lambda c: -c["score"])
        return k, kandidat, pilih_per_topik(kandidat, k)
    return k, kandidat, pilih_beragam(kandidat, k)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Embedding pertanyaan, lalu ambil potongan dokumen dari Qdrant (lihat ambil_konteks)."""
    try:
        return ambil_konteks(question, top_k)[2]
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Tidak bisa terhubung ke Qdrant ({config.QDRANT_URL}). "
            f"Periksa koneksi internet dan QDRANT_URL. Detail: {exc}"
        ) from exc


# Aturan tambahan untuk GAYA_JAWABAN=langsung (lihat config.py). Tidak dipakai
# saat evaluasi, kecuali GAYA_JAWABAN diatur di .env.
ATURAN_LANGSUNG = (
    " Kalimat pertama jawaban harus langsung menjawab inti pertanyaan, tanpa "
    "kalimat pembuka seperti 'Berikut adalah' dan tanpa mengulang pertanyaan. "
    "Setelah itu baru berikan rincian seperlunya. Kecuali saat informasi tidak "
    "ditemukan, jangan menutup jawaban dengan tawaran untuk bertanya lagi, karena "
    "aplikasi sudah menampilkan saran pertanyaan lanjutan."
)


def system_prompt_rag() -> str:
    """Instruksi sistem mode RAG sesuai GAYA_JAWABAN."""
    if config.GAYA_JAWABAN == "langsung":
        return SYSTEM_PROMPT + ATURAN_LANGSUNG
    return SYSTEM_PROMPT


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        page = f", hlm. {ctx['page']}" if ctx.get("page") else ""
        blocks.append(f"[{i}] (Sumber: {ctx['source']}{page})\n{ctx['text']}")
    return PROMPT_TEMPLATE.format(
        context="\n\n".join(blocks) if blocks else "(tidak ada konteks)",
        question=question,
    )


def answer(question: str, top_k: int | None = None, model: str | None = None,
           riwayat: list[dict] | None = None) -> dict:
    """Satu siklus penuh RAG: jawaban, konteks sumber, serta waktu tiap tahap.

    riwayat (opsional): giliran tanya-jawab sebelumnya. Bila ada, pertanyaan
    lebih dulu ditulis ulang menjadi pertanyaan mandiri sebelum retrieval.
    Evaluasi memanggil fungsi ini tanpa riwayat, jadi hasil evaluasi tidak
    terpengaruh fitur percakapan.

    "timing" (detik): rewrite_s = penulisan ulang pertanyaan (0 tanpa riwayat),
    retrieval_s = embedding kueri + pencarian Qdrant, generation_s = panggilan
    LLM, total_s = seluruh proses.
    """
    t0 = time.perf_counter()
    riwayat = rapikan_riwayat(riwayat)
    pertanyaan, rewrite_s = tulis_ulang_pertanyaan(question, riwayat, model=model)
    if not riwayat:
        # Pertanyaan pertama yang terpotong, misalnya "padi adalah", dilengkapi dengan aturan
        pertanyaan = lengkapi_pertanyaan(pertanyaan)
    t_awal = time.perf_counter()
    contexts = retrieve(pertanyaan, top_k=top_k)
    t1 = time.perf_counter()
    if not contexts:
        return {
            "answer": "Informasi tersebut tidak ditemukan pada basis pengetahuan.",
            "contexts": [],
            "pertanyaan_mandiri": pertanyaan,
            "timing": {"rewrite_s": rewrite_s, "retrieval_s": round(t1 - t_awal, 3),
                       "generation_s": 0.0, "total_s": round(t1 - t0, 3)},
            "usage": {},
        }
    hasil = llm.generate_detail(
        build_prompt(pertanyaan, contexts),
        system_prompt=system_prompt_rag(),
        model=model,
    )
    t2 = time.perf_counter()
    return {
        "answer": hasil["text"],
        "contexts": contexts,
        "pertanyaan_mandiri": pertanyaan,
        "timing": {
            "rewrite_s": rewrite_s,
            "retrieval_s": round(t1 - t_awal, 3),
            "generation_s": hasil["latency_s"],
            "total_s": round(t2 - t0, 3),
        },
        "usage": {k: hasil[k] for k in ("model", "prompt_tokens", "completion_tokens",
                                        "reasoning_tokens", "total_tokens")},
    }


def answer_tanpa_rag(question: str, model: str | None = None,
                     riwayat: list[dict] | None = None) -> dict:
    """Pembanding: pertanyaan langsung ke LLM tanpa retrieval (LLM saja).

    Struktur hasil sama dengan answer(), dengan contexts kosong dan
    retrieval_s = 0, sehingga kedua mode dapat dievaluasi dan dibandingkan.
    """
    t0 = time.perf_counter()
    hasil = llm.generate_detail(
        f"Pertanyaan pengguna:\n{question}\n\nJawaban:",
        system_prompt=SYSTEM_PROMPT_TANPA_RAG,
        model=model,
        riwayat=rapikan_riwayat(riwayat),  # tanpa retrieval, LLM cukup melihat riwayat
    )
    t1 = time.perf_counter()
    return {
        "answer": hasil["text"],
        "contexts": [],
        "pertanyaan_mandiri": question,
        "timing": {
            "rewrite_s": 0.0,
            "retrieval_s": 0.0,
            "generation_s": hasil["latency_s"],
            "total_s": round(t1 - t0, 3),
        },
        "usage": {k: hasil[k] for k in ("model", "prompt_tokens", "completion_tokens",
                                        "reasoning_tokens", "total_tokens")},
    }


# Nomor rujukan di jawaban: [1], [1, 3], atau gaya gpt-oss 【1】 / 【1†L5-L9】
POLA_RUJUKAN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]|【\s*(\d+)[^】]*】")


def _label_sumber(ctx: dict) -> str:
    return ctx["source"] + (f" hlm. {ctx['page']}" if ctx.get("page") else "")


def sumber_dikutip(answer: str, contexts: list[dict]) -> list[str]:
    """Sumber yang benar-benar dirujuk di jawaban, beserta nomornya.

    Contoh: jawaban mengutip [1] dan [3] -> ["[1] buku.pdf hlm. 4", "[3] jurnal.pdf hlm. 2"].
    Jawaban yang tidak mengutip apa pun (misalnya penolakan karena di luar
    topik padi) menghasilkan daftar kosong, sehingga halaman web tidak
    menampilkan sumber yang tidak dipakai. Untuk evaluasi, seluruh konteks
    hasil retrieval tetap tersedia di hasil["contexts"].
    """
    nomor = set()
    for daftar, tunggal in POLA_RUJUKAN.findall(answer or ""):
        for n in (daftar.split(",") if daftar else [tunggal]):
            if n.strip().isdigit():
                nomor.add(int(n))
    return [f"[{i}] {_label_sumber(contexts[i - 1])}"
            for i in sorted(nomor) if 1 <= i <= len(contexts)]


def format_sources(contexts: list[dict]) -> list[str]:
    """Daftar sumber unik untuk ditampilkan di bawah jawaban."""
    hasil, sudah = [], set()
    for ctx in contexts:
        label = ctx["source"] + (f" hlm. {ctx['page']}" if ctx.get("page") else "")
        if label not in sudah:
            sudah.add(label)
            hasil.append(label)
    return hasil

