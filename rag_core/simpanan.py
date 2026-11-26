"""Penyimpanan jawaban untuk kueri baku (aplikasi web).

Tujuannya: pertanyaan yang maknanya sama SELALU mendapat jawaban dan rujukan
yang sama, bukan hanya selama satu server menyala. Vercel menjalankan beberapa
server sekaligus dan mematikannya bila sepi, sehingga simpanan di memori saja
tidak cukup. Jawaban dicari berurutan di:

1. Berkas jawaban baku (rag_core/data/jawaban_baku.json), dibuat sekali dengan
   scripts/buat_jawaban_baku.py lalu ikut di-push. Tidak berubah sampai berkas
   itu dibuat ulang. Ini lapisan utama dan tidak butuh layanan tambahan.
2. Redis Upstash (opsional), bila UPSTASH_REDIS_REST_URL dan
   UPSTASH_REDIS_REST_TOKEN (atau KV_REST_API_URL dan KV_REST_API_TOKEN dari
   integrasi Vercel) diisi. Jawaban pertama untuk kueri baku yang belum ada di
   berkas disimpan di sini dan dipakai bersama oleh semua server.
3. Memori server (lapisan tercepat, hilang saat server dimatikan).

Kegagalan Redis tidak pernah menggagalkan jawaban; sistem tetap menjawab.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import requests

from . import config

BERKAS_JAWABAN = Path(__file__).resolve().parent / "data" / "jawaban_baku.json"
MAKS_MEMORI = 256

_MEMORI: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()


def kunci(kueri: str, model: str | None = None, top_k: int | None = None) -> str:
    return f"{config.CACHE_VERSI}|{model or config.LLM_MODEL}|{top_k or ''}|{kueri}"


# --- 1. Berkas jawaban baku ----------------------------------------------------

@lru_cache(maxsize=1)
def _berkas() -> dict:
    try:
        data = json.loads(BERKAS_JAWABAN.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Berkas lama (dibuat sebelum dokumen atau prompt berubah) tidak dipakai
    if str(data.get("versi", "1")) != config.CACHE_VERSI:
        return {}
    return data.get("jawaban", {})


def dari_berkas(kueri: str, model: str | None = None, top_k: int | None = None) -> dict | None:
    """Jawaban dari berkas, hanya untuk model dan top_k bawaan (seperti saat dibuat)."""
    if top_k or (model and model != config.LLM_MODEL):
        return None
    isi = _berkas().get(kueri)
    if not isi or not isi.get("answer"):
        return None
    return {"answer": isi["answer"], "contexts": isi.get("contexts", []),
            "usage": isi.get("usage", {})}


# --- 2. Redis Upstash (REST) ---------------------------------------------------

def _redis(perintah: list) -> object:
    if not (config.REDIS_URL and config.REDIS_TOKEN):
        return None
    try:
        r = requests.post(
            config.REDIS_URL,
            headers={"Authorization": f"Bearer {config.REDIS_TOKEN}"},
            json=perintah,
            timeout=3,
        )
        r.raise_for_status()
        return r.json().get("result")
    except (requests.RequestException, ValueError):
        return None


def _kunci_redis(k: str) -> str:
    return "tanyapadi:jawaban:" + hashlib.sha1(k.encode("utf-8")).hexdigest()


# --- Antarmuka -----------------------------------------------------------------

def ambil(kueri: str, model: str | None = None, top_k: int | None = None) -> dict | None:
    if not config.SIMPAN_JAWABAN:
        return None
    k = kunci(kueri, model, top_k)
    isi = _MEMORI.get(k)
    if isi:
        _MEMORI.move_to_end(k)
        return isi[1]
    data = dari_berkas(kueri, model, top_k)
    if data is None:
        mentah = _redis(["GET", _kunci_redis(k)])
        if isinstance(mentah, str):
            try:
                data = json.loads(mentah)
            except ValueError:
                data = None
    if data:
        _ke_memori(k, data)
    return data


def simpan(kueri: str, data: dict, model: str | None = None, top_k: int | None = None) -> None:
    if not config.SIMPAN_JAWABAN:
        return
    k = kunci(kueri, model, top_k)
    _ke_memori(k, data)
    perintah = ["SET", _kunci_redis(k), json.dumps(data, ensure_ascii=False)]
    if config.SIMPAN_JAWABAN_DETIK > 0:
        perintah += ["EX", config.SIMPAN_JAWABAN_DETIK]
    _redis(perintah)


def _ke_memori(k: str, data: dict) -> None:
    _MEMORI[k] = (time.time(), data)
    _MEMORI.move_to_end(k)
    while len(_MEMORI) > MAKS_MEMORI:
        _MEMORI.popitem(last=False)


def kosongkan_memori() -> None:
    _MEMORI.clear()
    _berkas.cache_clear()
