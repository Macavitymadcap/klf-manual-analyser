"""aggregation/clusters.py — Optional Qdrant cluster analysis for the aggregate report."""

import logging

from manual_analyser.aggregation.types import ClusterInfo
from manual_analyser.embedding.constants import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT

logger = logging.getLogger(__name__)


def fetch_clusters(track_ids: list[str]) -> list[ClusterInfo]:
    """
    Query Qdrant for cluster groupings of the given tracks.

    Returns empty list if Qdrant is unavailable or clustering fails —
    the report renders without cluster features.
    """
    try:
        return _run_cluster_query(track_ids)
    except Exception as exc:
        logger.warning("[aggregation] Qdrant cluster query failed: %s", exc)
        return []


def _run_cluster_query(track_ids: list[str]) -> list[ClusterInfo]:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)
    points = _fetch_points(client, track_ids)
    if len(points) < 2:
        return []
    return _cluster_by_groove(points)


def _fetch_points(client, track_ids: list[str]) -> list[dict]:
    """Fetch payloads for all track_ids from Qdrant."""
    results = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=None,
        with_payload=True,
        with_vectors=False,
        limit=len(track_ids) + 10,
    )
    return [p.payload for p in results[0] if p.payload.get("track_id") in track_ids]


def _cluster_by_groove(points: list[dict]) -> list[ClusterInfo]:
    """Simple clustering by groove_feel — groups tracks by their dominant feel."""
    groups: dict[str, list[str]] = {}
    for p in points:
        feel = p.get("groove_feel", "unknown")
        groups.setdefault(feel, []).append(p["track_id"])

    clusters = []
    for i, (feel, ids) in enumerate(groups.items()):
        clusters.append(
            ClusterInfo(
                cluster_id=i,
                track_ids=ids,
                dominant_features={"groove_feel": feel},
            )
        )
    return clusters
