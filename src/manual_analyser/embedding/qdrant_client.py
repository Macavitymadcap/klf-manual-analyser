"""embed/qdrant_client.py — Qdrant upsert and availability check."""

import uuid

from manual_analyser.embedding.constants import (
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    VECTOR_SIZE,
)


class QdrantUnavailableError(Exception):
    """Raised when Qdrant is not reachable. Stage is skipped gracefully."""


def check_qdrant() -> None:
    """Verify Qdrant is reachable. Raises QdrantUnavailableError if not."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
        client.get_collections()
    except Exception as exc:
        raise QdrantUnavailableError(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}: {exc}") from exc


def ensure_collection() -> None:
    """Create the tracks collection if it does not exist."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert(track_id: str, vector: list[float], payload: dict) -> str:
    """
    Upsert a vector into the tracks collection.

    Returns the Qdrant point UUID.
    Raises QdrantUnavailableError on failure.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    point_id = str(uuid.uuid4())
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id
    except Exception as exc:
        raise QdrantUnavailableError(f"Qdrant upsert failed: {exc}") from exc
