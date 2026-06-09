"""scoring/llm/ollama_client.py — Raw HTTP calls to Ollama's chat endpoint."""

import logging

import httpx

from .constants import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_PATH,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Raised when Ollama is not reachable. Hard abort for scoring stage."""


class OllamaTimeoutError(Exception):
    """Raised when a request exceeds REQUEST_TIMEOUT."""


def check_ollama(model: str) -> None:
    """
    Verify Ollama is running and the model is available.

    Raises OllamaUnavailableError on any failure.
    """
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        raise OllamaUnavailableError(f"Ollama not reachable at {OLLAMA_BASE_URL}. Start Ollama and retry.") from exc

    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(m.startswith(model.split(":")[0]) for m in models):
        raise OllamaUnavailableError(f"Model '{model}' not found. Run: ollama pull {model}")


def chat(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Send a single chat request to Ollama and return the response text.

    Raises:
        OllamaTimeoutError: if the request times out.
        httpx.HTTPError: on HTTP-level failures.
    """
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}{OLLAMA_CHAT_PATH}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except httpx.TimeoutException as exc:
        raise OllamaTimeoutError("Ollama request timed out") from exc
