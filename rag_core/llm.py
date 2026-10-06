"""Pemanggilan LLM melalui API pihak ketiga.

Endpoint DeepSeek kompatibel dengan format OpenAI Chat Completions, sehingga
fungsi generate() di bawah dapat dipakai ulang untuk penyedia lain yang
memakai format yang sama hanya dengan mengganti base_url, api_key, dan model.
Hal ini memudahkan eksperimen perbandingan beberapa LLM pada pipeline yang sama.
"""

import requests

from rag_core import config


def generate(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> str:
    model = model or config.LLM_MODEL
    base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
    api_key = api_key or config.LLM_API_KEY
    if not api_key:
        raise RuntimeError("LLM_API_KEY belum diatur.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": (
                config.LLM_TEMPERATURE if temperature is None else temperature
            ),
            "stream": False,
        },
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

