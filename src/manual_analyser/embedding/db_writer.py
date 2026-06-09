"""embedding/db_writer.py — Write embedding results back to SQLite."""

from pathlib import Path

from manual_analyser.db import get_connection


def write_feature_summary(track_id: str, summary: str, db_path: Path) -> None:
    """Persist the feature summary text to tracks.feature_summary."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE tracks SET feature_summary = ? WHERE track_id = ?",
                (summary, track_id),
            )
    finally:
        conn.close()


def write_vector_record(track_id: str, qdrant_id: str, db_path: Path) -> None:
    """Insert a track_vectors row linking track_id to its Qdrant point UUID."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO track_vectors (track_id, qdrant_id) VALUES (?, ?)",
                (track_id, qdrant_id),
            )
    finally:
        conn.close()
