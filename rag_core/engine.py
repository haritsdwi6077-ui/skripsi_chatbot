"""Mesin RAG: retrieval dari Qdrant Cloud, penyusunan prompt, lalu generation.

Versi sebelumnya memakai rantai LangChain dengan Gemini. Versi ini memanggil
API secara langsung agar dependensinya ringan (penting untuk deploy di Vercel)
dan agar model LLM mudah diganti untuk kebutuhan perbandingan antar-LLM.
"""

import json
import re
import time

from rag_core import config, embeddings, llm, saran, simpanan
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

# Prompt pembanding (LLM saja, tanpa retrieval). Ada dua varian:
#   "kosong" : LLM murni tanpa system prompt. Ini rancangan utama penelitian,
#              dipakai untuk glm_llm.csv (LLM bawaan vs sistem RAG lengkap).
#   "setara" : peran, bahasa, ranah, dan format jawaban dibuat sama dengan
#              SYSTEM_PROMPT. Aturan yang bergantung pada konteks (nomor sumber,
#              "jawab hanya dari konteks") tidak disertakan karena tidak ada
#              konteks. Varian ini menjadi kontrol, agar selisih skor dapat
#              dikaitkan dengan retrieval, bukan dengan perbedaan instruksi
#              atau panjang jawaban.
SYSTEM_PROMPT_TANPA_RAG = ""

SYSTEM_PROMPT_TANPA_RAG_SETARA = (
    "Anda adalah asisten penyuluh pertanian yang menjawab pertanyaan seputar "
    "hama dan penyakit tanaman padi. Jawaban selalu ditulis dalam bahasa "
    "Indonesia yang ringkas dan mudah dipahami petani. Jika Anda tidak "
    "mengetahui jawabannya, katakan terus terang dan jangan mengarang jawaban."
    + RANAH
    + FORMAT_JAWABAN
)

PROMPT_TANPA_RAG = {"kosong": SYSTEM_PROMPT_TANPA_RAG, "setara": SYSTEM_PROMPT_TANPA_RAG_SETARA}

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
           riwayat: list[dict] | None = None, percakapan: bool = False) -> dict:
    """Satu siklus penuh RAG: jawaban, konteks sumber, serta waktu tiap tahap.

    riwayat (opsional): giliran tanya-jawab sebelumnya. Bila ada, pertanyaan
    lebih dulu ditulis ulang menjadi pertanyaan mandiri sebelum retrieval.
    Evaluasi memanggil fungsi ini tanpa riwayat, jadi hasil evaluasi tidak
    terpengaruh fitur percakapan.

    percakapan=True dipakai aplikasi web: setiap pesan lebih dulu dipahami oleh
    LLM (lihat answer_percakapan). Bawaannya False, sehingga jalur yang dipakai
    scripts/evaluate.py tetap sama dengan jalur yang dievaluasi di skripsi.

    "timing" (detik): rewrite_s = penulisan ulang pertanyaan (0 tanpa riwayat),
    retrieval_s = embedding kueri + pencarian Qdrant, generation_s = panggilan
    LLM, total_s = seluruh proses.
    """
    if percakapan:
        return answer_percakapan(question, top_k=top_k, model=model, riwayat=riwayat)
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
        "answer": rapikan_sitasi(hasil["text"]),
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
                     riwayat: list[dict] | None = None, prompt: str = "kosong") -> dict:
    """Pembanding: pertanyaan langsung ke LLM tanpa retrieval (LLM saja).

    Struktur hasil sama dengan answer(), dengan contexts kosong dan
    retrieval_s = 0, sehingga kedua mode dapat dievaluasi dan dibandingkan.
    prompt: "kosong" (LLM murni, bawaan) atau "setara" (instruksi sama dengan
    mode RAG, lihat SYSTEM_PROMPT_TANPA_RAG_SETARA).
    """
    if prompt not in PROMPT_TANPA_RAG:
        raise ValueError(f"prompt harus salah satu dari {sorted(PROMPT_TANPA_RAG)}")
    t0 = time.perf_counter()
    hasil = llm.generate_detail(
        f"Pertanyaan pengguna:\n{question}\n\nJawaban:",
        system_prompt=PROMPT_TANPA_RAG[prompt],
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


# --- Mode percakapan (aplikasi web) ------------------------------------------
# Revisi seminar hasil: pengguna tidak boleh diarahkan ke pertanyaan template.
# Petani bisa menulis singkat, memakai bahasa sehari-hari, singkatan ("gmn",
# "knp"), salah ketik, atau hanya menceritakan gejala. Karena itu SETIAP pesan,
# bukan hanya pertanyaan lanjutan, lebih dulu dipahami oleh LLM dan diubah
# menjadi kueri pencarian yang jelas. Salam dan basa-basi dijawab wajar tanpa
# mencari dokumen. Jalur ini hanya dipakai aplikasi web (answer(...,
# percakapan=True)); evaluasi RAGAS tetap memakai jalur answer() bawaan.

SYSTEM_PROMPT_PAHAMI = (
    "Anda membantu aplikasi tanya jawab hama dan penyakit padi memahami pesan "
    "pengguna. Pengguna sering menulis singkat, memakai bahasa sehari-hari atau "
    "bahasa daerah, singkatan seperti gmn, knp, yg, atau sdh, salah ketik, dan "
    "kadang hanya menceritakan gejala tanpa menyebut nama hama atau penyakit.\n\n"
    "Tugas Anda:\n"
    "1. Tentukan jenis pesan. Pilih \"sapaan\" bila pesan hanya berisi salam, "
    "ucapan terima kasih, basa-basi, atau pertanyaan tentang aplikasi ini, "
    "misalnya \"halo\", \"makasih\", atau \"kamu bisa bantu apa\". Selain itu "
    "pilih \"pertanyaan\".\n"
    "2. Untuk jenis \"pertanyaan\", tulis ulang pesan menjadi satu kalimat tanya "
    "bahasa Indonesia baku yang lengkap dan bisa dipahami tanpa membaca "
    "percakapan sebelumnya. Perbaiki salah ketik dan singkatan. Ganti kata "
    "rujukan seperti \"itu\", \"tersebut\", atau akhiran \"-nya\" dengan hal "
    "yang dimaksud dari percakapan sebelumnya. Bila pengguna menceritakan "
    "gejala, tulis sebagai pertanyaan tentang kemungkinan penyebab gejala itu "
    "pada tanaman padi. Pertahankan maksud asli pengguna, jangan menambahkan "
    "nama hama atau penyakit yang tidak disebut atau tidak jelas tersirat, dan "
    "jangan menjawab pertanyaannya.\n"
    "3. Untuk jenis \"pertanyaan\", tentukan juga aspek yang ditanyakan, pilih "
    "SATU: pengertian, gejala, penyebab, pengendalian, pencegahan, musuh_alami, "
    "daftar_penyakit, daftar_hama, daftar_hama_penyakit, atau lainnya. Obat, "
    "cara membasmi, atau cara mengatasi termasuk pengendalian. Ciri atau tanda "
    "serangan termasuk gejala. Pertanyaan tentang penyakit apa saja yang ada "
    "pada padi termasuk daftar_penyakit, hama apa saja termasuk daftar_hama, "
    "dan hama serta penyakit atau masalah padi secara umum termasuk "
    "daftar_hama_penyakit. Pertanyaan yang membandingkan atau menceritakan "
    "gejala tanpa nama hama atau penyakit termasuk lainnya.\n"
    "4. Isi \"topik\" dengan nama hama atau penyakit padi yang dimaksud, persis "
    "seperti yang disebut atau dirujuk pengguna. Kosongkan bila tidak ada.\n\n"
    "Balas hanya dengan JSON satu baris, tanpa teks lain: "
    "{\"jenis\": \"pertanyaan\", \"aspek\": \"...\", \"topik\": \"...\", \"kueri\": \"...\"} atau "
    "{\"jenis\": \"sapaan\", \"aspek\": \"\", \"topik\": \"\", \"kueri\": \"\"}"
)

# Kueri baku untuk pencarian dokumen. Pertanyaan yang artinya sama, misalnya
# "wereng coklat gmn ngatasinya" dan "cara basmi WBC", disusun menjadi kalimat
# pencarian yang PERSIS sama, sehingga potongan dokumen yang diambil juga sama.
# Pola ini hanya dipakai di dalam sistem dan tidak ditampilkan ke pengguna.
KUERI_BAKU = {
    "pengertian": {"penyakit": "Apa itu {nama} pada tanaman padi?",
                   "hama": "Apa itu hama {nama} pada tanaman padi?"},
    "gejala": {"penyakit": "Apa gejala {nama} pada tanaman padi?",
               "hama": "Apa gejala serangan {nama} pada tanaman padi?"},
    "penyebab": {"penyakit": "Apa penyebab {nama} pada tanaman padi?",
                 "hama": "Apa penyebab serangan {nama} pada tanaman padi?"},
    "pengendalian": {"penyakit": "Bagaimana cara mengendalikan {nama} pada tanaman padi?",
                     "hama": "Bagaimana cara mengendalikan {nama} pada tanaman padi?"},
    "pencegahan": {"penyakit": "Bagaimana cara mencegah {nama} pada tanaman padi?",
                   "hama": "Bagaimana cara mencegah serangan {nama} pada tanaman padi?"},
    "musuh_alami": {"penyakit": None,
                    "hama": "Apa saja musuh alami {nama} pada tanaman padi?"},
}


# Pertanyaan daftar ("apa penyakit pada padi", "penyakit padi apa aja") tidak
# menyebut satu nama, jadi memakai kalimat pencarian baku tersendiri.
KUERI_DAFTAR = {
    "daftar_penyakit": "Apa saja penyakit yang menyerang tanaman padi?",
    "daftar_hama": "Apa saja hama yang menyerang tanaman padi?",
    "daftar_hama_penyakit": "Apa saja hama dan penyakit yang menyerang tanaman padi?",
}


def kueri_baku(aspek: str, topik: str, kueri: str) -> str:
    """Kalimat pencarian baku bila aspek dan tepat satu hama/penyakit dikenali.

    Nama hama/penyakit diseragamkan lewat saran.cari_topik(), misalnya
    "wereng coklat", "WBC", dan "brown planthopper" menjadi
    "wereng batang cokelat". Bila tidak bisa dibakukan (aspek lainnya, nama
    tidak dikenali, atau lebih dari satu nama), kueri dari LLM dipakai apa adanya.
    """
    aspek = (aspek or "").strip().lower()
    if aspek in KUERI_DAFTAR:
        return KUERI_DAFTAR[aspek]
    pola = KUERI_BAKU.get(aspek)
    if not pola:
        return kueri
    dikenali = saran.cari_topik(topik) or saran.cari_topik(kueri)
    nama_unik = {nama for nama, _ in dikenali}
    if len(nama_unik) != 1:
        return kueri
    nama, jenis = dikenali[0]
    templat = pola.get(jenis)
    if not templat:
        return kueri
    return templat.format(nama=nama)

SYSTEM_PROMPT_SAPAAN = (
    "Anda adalah Tanya Padi, asisten yang membantu petani memahami hama dan "
    "penyakit tanaman padi. Balas pesan pengguna dengan ramah dan singkat (1-2 "
    "kalimat) dalam bahasa Indonesia sehari-hari. Bila sesuai, sampaikan bahwa "
    "pengguna boleh langsung bertanya atau menceritakan apa yang terlihat di "
    "sawah dengan bahasanya sendiri. Jangan memberi informasi teknis tentang "
    "hama atau penyakit, dan jangan memberi daftar contoh pertanyaan."
)

SYSTEM_PROMPT_PERCAKAPAN = (
    "Anda adalah Tanya Padi, asisten penyuluh pertanian yang membantu petani "
    "memahami hama dan penyakit tanaman padi. Pengguna bisa bertanya singkat, "
    "memakai bahasa sehari-hari, singkatan, atau salah ketik. Pahami maksudnya "
    "dan jawab langsung seperti sedang berbicara dengan petani, tanpa meminta "
    "pengguna mengubah cara bertanya. Isi jawaban HANYA boleh diambil dari "
    "konteks yang diberikan. Dokumen konteks dapat berbahasa Indonesia maupun "
    "Inggris, tetapi jawaban selalu ditulis dalam bahasa Indonesia yang "
    "sederhana. Sesuaikan panjang jawaban dengan pertanyaan: pertanyaan singkat "
    "dijawab singkat dan langsung ke inti, lalu rincian seperlunya. Jika "
    "pengguna menceritakan gejala tanpa menyebut nama hama atau penyakit, "
    "sebutkan kemungkinan penyebab yang sesuai dengan konteks beserta ciri "
    "pembedanya, lalu boleh tanyakan satu hal singkat untuk memastikan, "
    "misalnya bagian tanaman yang terserang. Jika konteks tidak memuat "
    "informasi yang ditanyakan, katakan terus terang bahwa informasinya belum "
    "ada di dokumen rujukan, dan jangan mengarang jawaban. Jika pertanyaan "
    "meminta daftar atau gambaran umum, sebutkan hama atau penyakit yang ada di "
    "konteks. Cantumkan nomor sumber, misalnya [1], pada bagian jawaban yang "
    "mengambil informasi dari konteks tersebut. Jangan menambahkan hama, "
    "penyakit, atau keterangan yang tidak ada di konteks. Jangan menebak "
    "kepanjangan singkatan yang tidak dijelaskan di konteks. Bila sebagian "
    "konteks membahas tanaman lain seperti jagung atau kedelai, abaikan bagian "
    "itu dan jawab hanya untuk tanaman padi."
    + RANAH
    + FORMAT_JAWABAN
)

PROMPT_TEMPLATE_PERCAKAPAN = """Konteks:
{context}

Pesan pengguna:
{asli}

Maksud pesan menurut sistem:
{kueri}

Jawaban:"""

JENIS_PESAN = ("pertanyaan", "sapaan")


def _baca_json(teks: str) -> dict | None:
    """Ambil objek JSON pertama dari jawaban LLM (kadang dibungkus ```json)."""
    m = re.search(r"\{.*?\}", teks or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def semua_kueri_baku() -> list[str]:
    """Semua kalimat pencarian baku: 3 kueri daftar + setiap hama/penyakit x aspek."""
    hasil = list(KUERI_DAFTAR.values())
    for nama, jenis, _ in saran.TOPIK:
        for pola in KUERI_BAKU.values():
            templat = pola.get(jenis)
            if templat:
                hasil.append(templat.format(nama=nama))
    return hasil


def adalah_kueri_baku(kueri: str) -> bool:
    """True bila kueri adalah salah satu kalimat pencarian baku (KUERI_BAKU/DAFTAR)."""
    return kueri in semua_kueri_baku()


# --- Pemahaman berbasis aturan (tanpa LLM) ----------------------------------
# Pertanyaan yang umum, yaitu pertanyaan daftar ("apa penyakit padi", "hama
# padi apa aja") dan pertanyaan satu hama/penyakit + satu aspek ("wbc gmn
# ngatasinya", "gejala blas"), dikenali dengan aturan tetap. Hasilnya selalu
# sama untuk kalimat yang maknanya sama, tidak bergantung pada tebakan LLM.
# Aturan hanya dipakai bila SEMUA kata dikenali; selain itu diserahkan ke LLM.

KATA_PENGISI = set("""
padi tanaman sawah pada di ke yang yg menyerang diserang serang terserang kena
apa apaan apakah apa2 saja aja sajakah sih ya dong deh kak min bang pak bu nih
sebutkan sebut jenis jenis2 macam macam2 daftar umum utama ada berapa terdapat
itu ini dan serta atau bisa biasanya sering tolong mau tanya saya aku sy gw gue
jelaskan jelasin kasih tau tahu info informasi contoh contohnya biasa banyak
bagaimana gimana gmn gmna bgmn bagaimanakah cara caranya untuk utk buat agar
supaya serangan hama penyakit nya kah dengan dgn tentang ttg mengenai the
""".split())

# Penggalan kata penanda aspek (dicocokkan sebagai bagian kata, sehingga
# "ngatasinya", "mengendalikan", dan "obatnya" ikut dikenali).
STEM_ASPEK = (
    ("musuh_alami", ("musuh", "predator", "parasitoid")),
    ("pencegahan", ("cegah", "antisipasi", "hindar")),
    ("pengendalian", ("endali", "atasi", "basmi", "berantas", "obat", "insektisida",
                      "fungisida", "bakterisida", "pestisida", "semprot", "usir",
                      "tangan", "solusi", "racun")),
    ("penyebab", ("sebab", "nyebab", "patogen", "kenapa", "mengapa", "knp")),
    ("gejala", ("gejala", "tanda", "ciri", "seperti", "kayak", "kaya")),
    ("pengertian", ("pengertian", "definisi", "maksud", "arti")),
)
KATA_TAMBAHAN_ASPEK = {"alami"}  # bagian dari "musuh alami"


def _token(teks: str) -> list[str]:
    teks = (teks or "").lower().replace("coklat", "cokelat")
    teks = re.sub(r"\b(\w+)-\1\b", r"\1", teks)  # jenis-jenis -> jenis
    hasil = []
    for t in re.findall(r"[a-z0-9]+", teks):
        t = re.sub(r"2$", "", t) if len(t) > 3 else t   # hama2 -> hama
        if t.endswith("nya") and t[:-3] in KATA_PENGISI:  # penyakitnya -> penyakit
            t = t[:-3]
        hasil.append(t)
    return hasil


def _aspek_kata(kata: str) -> str | None:
    for aspek, stem in STEM_ASPEK:
        if any(s in kata for s in stem):
            return aspek
    return None


def pahami_aturan(teks: str) -> str | None:
    """Kueri baku dari aturan tetap, atau None bila pesan tidak cukup sederhana."""
    teks = (teks or "").lower().replace("coklat", "cokelat")
    dikenali = saran.cari_topik(teks)
    nama_unik = {nama for nama, _ in dikenali}

    if not nama_unik:
        # Pertanyaan daftar: hanya berisi kata "penyakit" dan/atau "hama" + kata pengisi
        kata = _token(teks)
        sisa = [k for k in kata if k not in KATA_PENGISI]
        if sisa:
            return None
        ada_penyakit, ada_hama = "penyakit" in kata, "hama" in kata
        if ada_penyakit and ada_hama:
            return KUERI_DAFTAR["daftar_hama_penyakit"]
        if ada_penyakit:
            return KUERI_DAFTAR["daftar_penyakit"]
        if ada_hama:
            return KUERI_DAFTAR["daftar_hama"]
        return None

    if len(nama_unik) != 1:
        return None
    nama, jenis = dikenali[0]
    # Buang nama hama/penyakit dari kalimat, lalu periksa sisa katanya
    for _nama, _jenis, pola in saran._TOPIK:  # noqa: SLF001
        if _nama == nama:
            teks = pola.sub(" ", teks)
    aspek_ditemukan = set()
    for k in _token(teks):
        if k in KATA_PENGISI or k in KATA_TAMBAHAN_ASPEK:
            continue
        aspek = _aspek_kata(k)
        if not aspek:
            return None  # ada kata yang tidak dikenali: serahkan ke LLM
        aspek_ditemukan.add(aspek)
    if len(aspek_ditemukan) > 1:
        return None
    aspek = aspek_ditemukan.pop() if aspek_ditemukan else "pengertian"
    templat = KUERI_BAKU.get(aspek, {}).get(jenis)
    return templat.format(nama=nama) if templat else None


def pahami_pertanyaan(question: str, riwayat: list[dict],
                      model: str | None = None) -> tuple[str, str, float]:
    """Memahami satu pesan pengguna sebelum retrieval.

    Mengembalikan (jenis, kueri, detik). jenis = "pertanyaan" atau "sapaan".
    Bila LLM gagal atau hasilnya janggal, dipakai cara lama sebagai cadangan:
    tulis_ulang_pertanyaan() bila ada riwayat, selain itu lengkapi_pertanyaan().
    """
    def cadangan() -> tuple[str, str, float]:
        if riwayat:
            kueri, detik = tulis_ulang_pertanyaan(question, riwayat, model=model)
            return "pertanyaan", kueri, detik
        return "pertanyaan", lengkapi_pertanyaan(question), 0.0

    # 1. Aturan tetap dulu: hasilnya sama untuk kalimat yang maknanya sama
    kueri_aturan = pahami_aturan(question)
    if kueri_aturan:
        return "pertanyaan", kueri_aturan, 0.0

    percakapan = "\n\n".join(
        f"Pengguna: {g['question']}\nAsisten: {g['answer']}" for g in riwayat
    )
    prompt = (f"Percakapan sebelumnya:\n{percakapan}\n\n" if percakapan else "") + \
        f"Pesan pengguna: {question}"
    try:
        hasil = llm.generate_detail(
            prompt,
            system_prompt=SYSTEM_PROMPT_PAHAMI,
            model=model,
            temperature=0,
            reasoning_effort="low" if config.LLM_REASONING_EFFORT else "",
            seed=config.LLM_SEED,
        )
    except Exception:  # noqa: BLE001
        return cadangan()
    data = _baca_json(hasil["text"])
    if not data:
        return cadangan()
    jenis = str(data.get("jenis") or "").strip().lower()
    kueri = re.sub(r"\s+", " ", str(data.get("kueri") or "")).strip()
    if jenis == "sapaan":
        return "sapaan", "", hasil["latency_s"]
    if jenis != "pertanyaan" or not kueri or len(kueri) > 400:
        return cadangan()
    kueri = kueri_baku(str(data.get("aspek") or ""), str(data.get("topik") or ""), kueri)
    if not adalah_kueri_baku(kueri):
        # 2. Jaring pengaman: LLM kadang salah memilih aspek, padahal kalimat
        # tulis ulangnya sederhana, misalnya "Apa saja penyakit yang menyerang
        # padi?". Kalimat itu diperiksa lagi dengan aturan tetap.
        kueri = pahami_aturan(kueri) or kueri
    return "pertanyaan", kueri, hasil["latency_s"]


def build_prompt_percakapan(asli: str, kueri: str, contexts: list[dict]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        page = f", hlm. {ctx['page']}" if ctx.get("page") else ""
        blocks.append(f"[{i}] (Sumber: {ctx['source']}{page})\n{ctx['text']}")
    return PROMPT_TEMPLATE_PERCAKAPAN.format(
        context="\n\n".join(blocks) if blocks else "(tidak ada konteks)",
        asli=asli,
        kueri=kueri,
    )


def _usage(hasil: dict) -> dict:
    return {k: hasil.get(k) for k in ("model", "prompt_tokens", "completion_tokens",
                                      "reasoning_tokens", "total_tokens")}


def answer_percakapan(question: str, top_k: int | None = None, model: str | None = None,
                      riwayat: list[dict] | None = None) -> dict:
    """Jalur aplikasi web: pahami pesan, lalu jawab seperti asisten AI pada umumnya.

    Struktur hasil sama dengan answer(), ditambah kunci "jenis"
    ("pertanyaan" atau "sapaan") dan "dari_cache".

    Bila pesan dipahami sebagai kueri baku (misalnya "apa penyakit pada padi"
    atau "wereng coklat gmn ngatasinya"), jawaban disusun HANYA dari kueri baku
    itu: tanpa riwayat obrolan dan tanpa kata-kata asli pengguna, lalu disimpan.
    Dengan begitu pertanyaan yang maknanya sama mendapat jawaban dan rujukan yang
    sama. Pertanyaan lain (cerita gejala, perbandingan, lanjutan yang tidak
    baku) tetap dijawab dengan riwayat agar obrolannya nyambung.
    """
    t0 = time.perf_counter()
    riwayat = rapikan_riwayat(riwayat)
    jenis, kueri, rewrite_s = pahami_pertanyaan(question, riwayat, model=model)

    if jenis == "sapaan":
        hasil = llm.generate_detail(
            question, system_prompt=SYSTEM_PROMPT_SAPAAN, model=model, riwayat=riwayat,
        )
        t1 = time.perf_counter()
        return {
            "answer": hasil["text"],
            "contexts": [],
            "pertanyaan_mandiri": question,
            "jenis": "sapaan",
            "timing": {"rewrite_s": rewrite_s, "retrieval_s": 0.0,
                       "generation_s": hasil["latency_s"], "total_s": round(t1 - t0, 3)},
            "usage": _usage(hasil),
        }

    baku = adalah_kueri_baku(kueri)
    if baku:
        simpanan_ = simpanan.ambil(kueri, model, top_k)
        if simpanan_:
            return {
                **simpanan_,
                "contexts": list(simpanan_["contexts"]),
                "pertanyaan_mandiri": kueri,
                "jenis": "pertanyaan",
                "dari_cache": True,
                "timing": {"rewrite_s": rewrite_s, "retrieval_s": 0.0, "generation_s": 0.0,
                           "total_s": round(time.perf_counter() - t0, 3)},
            }

    t_awal = time.perf_counter()
    contexts = retrieve(kueri, top_k=top_k)
    t1 = time.perf_counter()
    if not contexts:
        return {
            "answer": "Maaf, informasi tentang hal itu belum ada di dokumen rujukan Tanya Padi.",
            "contexts": [],
            "pertanyaan_mandiri": kueri,
            "jenis": "pertanyaan",
            "timing": {"rewrite_s": rewrite_s, "retrieval_s": round(t1 - t_awal, 3),
                       "generation_s": 0.0, "total_s": round(t1 - t0, 3)},
            "usage": {},
        }
    hasil = llm.generate_detail(
        # Kueri baku: hanya kueri itu yang dikirim, agar jawaban tidak bergantung
        # pada susunan kata pengguna maupun jawaban sebelumnya
        build_prompt_percakapan(kueri if baku else question, kueri, contexts),
        system_prompt=SYSTEM_PROMPT_PERCAKAPAN,
        model=model,
        riwayat=[] if baku else riwayat,
        # Temperature rendah dan seed tetap agar jawaban konsisten
        temperature=config.LLM_TEMPERATURE_PERCAKAPAN,
        seed=config.LLM_SEED,
    )
    t2 = time.perf_counter()
    jawaban = {
        "answer": rapikan_sitasi(hasil["text"]),
        "contexts": contexts,
        "usage": _usage(hasil),
    }
    if baku:
        simpanan.simpan(kueri, jawaban, model, top_k)
    return {
        **jawaban,
        "pertanyaan_mandiri": kueri,
        "jenis": "pertanyaan",
        "dari_cache": False,
        "timing": {
            "rewrite_s": rewrite_s,
            "retrieval_s": round(t1 - t_awal, 3),
            "generation_s": hasil["latency_s"],
            "total_s": round(t2 - t0, 3),
        },
    }


# Nomor rujukan di jawaban: [1], [1, 3], atau gaya gpt-oss 【1】 / 【1†L5-L9】
POLA_RUJUKAN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]|【\s*(\d+)[^】]*】")
# gpt-oss kadang memakai gaya kutipan bawaannya 【1】 atau 【1†L1-L4】 walaupun
# prompt meminta [1]. Diseragamkan agar tampilan dan teks yang dinilai konsisten.
POLA_SITASI_GPT = re.compile(r"【\s*(\d+)[^】]*】")


def rapikan_sitasi(teks: str) -> str:
    """Ubah 【1】 / 【1†L1-L4】 menjadi [1]."""
    return POLA_SITASI_GPT.sub(r"[\1]", teks or "")


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


def sumber_dikutip_rinci(answer: str, contexts: list[dict]) -> list[dict]:
    """Seperti sumber_dikutip(), tetapi berisi judul, halaman, dan tautan yang
    bisa diklik untuk halaman web (lihat rag_core/sumber.py)."""
    from rag_core import sumber  # impor lokal agar evaluasi tidak bergantung pada modul ini

    nomor = set()
    for daftar, tunggal in POLA_RUJUKAN.findall(answer or ""):
        for n in (daftar.split(",") if daftar else [tunggal]):
            if n.strip().isdigit():
                nomor.add(int(n))
    return [sumber.rincian(i, contexts[i - 1])
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

