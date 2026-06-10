"""pipeline/cache.py — Cache checks to decide which stages to skip."""

from pathlib import Path

from manual_analyser.db import get_connection


def track_in_db(track_id: str, db_path: Path) -> bool:
    """Return True if a tracks row exists for this track_id."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT 1 FROM tracks WHERE track_id = ? AND bpm IS NOT NULL", (track_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


def transcript_in_db(track_id: str, db_path: Path) -> bool:
    """Return True if transcript_segments rows exist for this track."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM transcript_segments WHERE track_id = ? LIMIT 1",
            (track_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def sections_labelled(track_id: str, db_path: Path) -> bool:
    """Return True if sections exist and at least one has a non-unknown label."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sections WHERE track_id = ? AND label != 'unknown' LIMIT 1",
            (track_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def vector_in_qdrant(track_id: str, db_path: Path) -> bool:
    """Return True if a track_vectors row exists (embedding was done)."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT 1 FROM track_vectors WHERE track_id = ?", (track_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


def scores_exist(track_id: str, mode: str, db_path: Path) -> bool:
    """Return True if any scores exist for this track + mode combination."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM scores WHERE track_id = ? AND mode = ? LIMIT 1",
            (track_id, mode),
        ).fetchone()
        return row is not None
    finally:
        conn.close()
