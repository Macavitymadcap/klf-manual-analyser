"""
db/helpers.py — Convenience query helpers used across the codebase.

All functions accept an open connection and return a simple Python value.
None of them open or close connections themselves — that is the caller's
responsibility.

Import via the package:
    from manual_analyser.db import get_track, track_exists, ...
"""

import sqlite3


def track_exists(conn: sqlite3.Connection, track_id: str) -> bool:
    """Return True if a tracks row exists for the given track_id."""
    row = conn.execute("SELECT 1 FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
    return row is not None


def transcript_exists(conn: sqlite3.Connection, track_id: str) -> bool:
    """Return True if any transcript segments exist for this track."""
    row = conn.execute("SELECT 1 FROM transcript_segments WHERE track_id = ? LIMIT 1", (track_id,)).fetchone()
    return row is not None


def sections_labelled(conn: sqlite3.Connection, track_id: str) -> bool:
    """
    Return True if the alignment pass has run for this track —
    i.e. at least one section has a label other than 'unknown'.
    """
    row = conn.execute(
        "SELECT 1 FROM sections WHERE track_id = ? AND label != 'unknown' LIMIT 1",
        (track_id,),
    ).fetchone()
    return row is not None


def vector_exists(conn: sqlite3.Connection, track_id: str) -> bool:
    """Return True if a Qdrant vector record exists for this track."""
    row = conn.execute("SELECT 1 FROM track_vectors WHERE track_id = ?", (track_id,)).fetchone()
    return row is not None


def scores_exist(conn: sqlite3.Connection, track_id: str, mode: str) -> bool:
    """Return True if scoring has been run for this track + mode combination."""
    row = conn.execute(
        "SELECT 1 FROM scores WHERE track_id = ? AND mode = ? LIMIT 1",
        (track_id, mode),
    ).fetchone()
    return row is not None


def get_section_sequence(conn: sqlite3.Connection, track_id: str) -> list[str]:
    """Return the ordered list of section labels for a track."""
    rows = conn.execute(
        "SELECT label FROM sections WHERE track_id = ? ORDER BY position",
        (track_id,),
    ).fetchall()
    return [row["label"] for row in rows]


def get_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    """Return the full tracks row for a track_id, or None if not found."""
    return conn.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,)).fetchone()


def get_all_track_ids(conn: sqlite3.Connection) -> list[str]:
    """Return all track_ids in the database."""
    rows = conn.execute("SELECT track_id FROM tracks").fetchall()
    return [row["track_id"] for row in rows]
