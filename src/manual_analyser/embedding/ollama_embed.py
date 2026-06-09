"""embedding/ollama_embed.py — Call nomic-embed-text via Ollama to get a vector."""

import httpx

from manual_analyser.embedding.constants import EMBED_MODEL, OLLAMA_BASE_URL


class EmbedUnavailableError(Exception):
    """Raised when the embedding model is not reachable. Skips stage for all tracks."""


def get_vector(text: str) -> list[float]:
    """
    Embed text via nomic-embed-text through Ollama.

    Returns a 384-dimensional float vector.
    Raises EmbedUnavailableError on connection or model failure.
    """
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except httpx.ConnectError as exc:
        raise EmbedUnavailableError(f"Ollama not reachable: {exc}") from exc
    except (KeyError, IndexError) as exc:
        raise EmbedUnavailableError(f"Unexpected embed response: {exc}") from exc


def check_embed_model() -> None:
    """Verify nomic-embed-text is available. Raises EmbedUnavailableError if not."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        raise EmbedUnavailableError(f"Ollama not reachable: {exc}") from exc

    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(m.startswith("nomic-embed-text") for m in models):
        raise EmbedUnavailableError("nomic-embed-text not found. Run: ollama pull nomic-embed-text")
