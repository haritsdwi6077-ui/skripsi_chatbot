"""Pemanggilan LLM melalui API pihak ketiga.

Endpoint Groq (dan DeepSeek, OpenAI, dll.) kompatibel dengan format OpenAI Chat
Completions, sehingga fungsi di bawah dapat dipakai untuk penyedia lain yang
memakai format yang sama hanya dengan mengganti base_url, api_key, dan model.

generate_detail() juga mencatat waktu respons dan jumlah token, untuk
melaporkan kecepatan sistem pada bab pengujian.
"""

import time

import requests

from rag_core import config


def generate_detail(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Kirim prompt ke LLM; kembalikan jawaban beserta waktu dan pemakaian token.

    Hasil: {"text", "model", "latency_s", "prompt_tokens", "completion_tokens",
            "reasoning_tokens", "total_tokens"}
    Nilai token bernilai None bila penyedia tidak melaporkannya.
    """
    model = model or config.LLM_MODEL
    base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
    api_key = api_key or config.LLM_API_KEY
    if reasoning_effort is None:
        reasoning_effort = config.LLM_REASONING_EFFORT
    if not api_key:
        raise RuntimeError("LLM_API_KEY belum diatur.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "stream": False,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if config.LLM_MAX_TOKENS > 0:
        body["max_tokens"] = config.LLM_MAX_TOKENS

    mulai = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=config.REQUEST_TIMEOUT,
    )
    latency = time.perf_counter() - mulai
    if not response.ok:
        raise RuntimeError(f"LLM {response.status_code} (model={model}): {response.text[:500]}")

    data = response.json()
    usage = data.get("usage") or {}
    detail = usage.get("completion_tokens_details") or {}
    return {
        "text": data["choices"][0]["message"].get("content") or "",
        "model": model,
        "latency_s": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": detail.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def generate(prompt: str, system_prompt: str = "", **kwargs) -> str:
    """Versi ringkas: hanya mengembalikan teks jawaban."""
    return generate_detail(prompt, system_prompt=system_prompt, **kwargs)["text"]
