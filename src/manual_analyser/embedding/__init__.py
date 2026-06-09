"""
embedding/embed.py — Stage 5: embed a track's features into Qdrant.

Public API:
    embed_track(track_id, db_path) -> EmbedResult | EmbedSkipped
    is_qdrant_available() -> bool

This stage is optional. If Qdrant or nomic-embed-text is unavailable,
the stage is skipped gracefully and a warning is logged once per run.
Pipeline.py calls is_qdrant_available() once at startup; if False it
skips all embed_track calls.
"""

import logging
from pathlib import Path

from manual_analyser.embedding import ollama_embed, qdrant_client
from manual_analyser.embedding.db_reader import load_track_features
from manual_analyser.embedding.db_writer import write_feature_summary, write_vector_record
from manual_analyser.embedding.ollama_embed import EmbedUnavailableError
from manual_analyser.embedding.payload import build_payload
from manual_analyser.embedding.qdrant_client import QdrantUnavailableError
from manual_analyser.embedding.summarise import build_summary
from manual_analyser.embedding.types import EmbedResult, EmbedSkipped

logger = logging.getLogger(__name__)

__all__ = ["embed_track", "is_qdrant_available"]


def is_qdrant_available() -> bool:
    """Return True if both Qdrant and nomic-embed-text are reachable."""
    try:
        qdrant_client.check_qdrant()
        ollama_embed.check_embed_model()
        return True
    except (QdrantUnavailableError, EmbedUnavailableError) as exc:
        logger.info("[embedding] Skipping embedding stage: %s", exc)
        return False


def embed_track(track_id: str, db_path: Path) -> EmbedResult | EmbedSkipped:
    """
    Embed one track and upsert to Qdrant.

    Returns EmbedResult on success, EmbedSkipped on any failure.
    Never raises.
    """
    short = track_id[:8]
    try:
        return _run(track_id, db_path, short)
    except (QdrantUnavailableError, EmbedUnavailableError) as exc:
        logger.warning("[%s] [embedding] Skipped: %s", short, exc)
        return EmbedSkipped(track_id=track_id, reason=str(exc))
    except Exception as exc:
        logger.exception("[%s] [embedding] Unexpected failure: %s", short, exc, exc_info=True)
        return EmbedSkipped(track_id=track_id, reason=f"unexpected: {exc}")


def _run(track_id: str, db_path: Path, short: str) -> EmbedResult:
    features = load_track_features(track_id, db_path)
    if features is None:
        raise ValueError(f"Track {short} not found in DB")

    summary = build_summary(features)
    write_feature_summary(track_id, summary, db_path)

    vector = ollama_embed.get_vector(summary)
    payload = build_payload(features)

    qdrant_client.ensure_collection()
    qdrant_id = qdrant_client.upsert(track_id, vector, payload)
    write_vector_record(track_id, qdrant_id, db_path)

    logger.info("[%s] [embedding] Embedded — qdrant_id=%s", short, qdrant_id)
    return EmbedResult(track_id=track_id, qdrant_id=qdrant_id, feature_summary=summary)
