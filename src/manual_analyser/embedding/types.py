"""embedding/types.py — Result types for the embedding stage."""

from dataclasses import dataclass


@dataclass
class EmbedResult:
    """Successful embedding for one track."""

    track_id: str
    qdrant_id: str
    feature_summary: str


@dataclass
class EmbedSkipped:
    """Embedding was skipped — Qdrant or nomic unavailable."""

    track_id: str
    reason: str
