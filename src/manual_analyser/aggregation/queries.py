"""aggregation/queries.py — SQL aggregation queries against the scores and tracks tables."""

from pathlib import Path

from manual_analyser.db import get_connection


def fetch_scored_track_ids(mode: str, db_path: Path) -> list[str]:
    """Return track_ids that have at least one score for this mode."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT track_id FROM scores WHERE mode = ?", (mode,)).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def fetch_criterion_ids(mode: str, db_path: Path) -> list[str]:
    """Return all criterion_ids scored for this mode."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT criterion_id FROM scores WHERE mode = ?", (mode,)).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def fetch_criterion_stats(criterion_id: str, mode: str, db_path: Path) -> dict:
    """Return pass_rate, mean_score, and scored_track_count for one criterion."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """SELECT
                AVG(CASE WHEN passed = 1 THEN 1.0 ELSE 0.0 END),
                AVG(score),
                COUNT(score)
               FROM scores
               WHERE mode = ? AND criterion_id = ? AND score IS NOT NULL""",
            (mode, criterion_id),
        ).fetchone()
        return {
            "pass_rate": row[0] or 0.0,
            "mean_score": row[1] or 0.0,
            "scored_track_count": row[2] or 0,
        }
    finally:
        conn.close()


def fetch_track_score_summary(track_id: str, mode: str, db_path: Path) -> dict:
    """Return overall_score, passed_count, total_count for one track."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """SELECT
                AVG(score),
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END),
                COUNT(*)
               FROM scores
               WHERE track_id = ? AND mode = ? AND score IS NOT NULL""",
            (track_id, mode),
        ).fetchone()
        return {
            "overall_score": row[0] or 0.0,
            "passed_count": row[1] or 0,
            "total_count": row[2] or 0,
        }
    finally:
        conn.close()


def fetch_track_metadata(track_id: str, db_path: Path) -> dict:
    """Return artist and song_name for one track."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT artist, song_name FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
        return {"artist": row[0], "song_name": row[1]} if row else {}
    finally:
        conn.close()


def fetch_modal_track_features(db_path: Path) -> dict:
    """Return the most common value for each key scalar feature across all tracks."""
    conn = get_connection(db_path)
    try:
        return {
            "modal_bpm": _modal_real(conn, "bpm"),
            "modal_key": _modal_text(conn, "key"),
            "modal_mode": _modal_text(conn, "mode"),
            "modal_groove_feel": _modal_text(conn, "groove_feel"),
            "modal_energy_shape": _modal_text(conn, "energy_shape"),
        }
    finally:
        conn.close()


def fetch_modal_structure(db_path: Path) -> list[str]:
    """Return the most common section label sequence across all tracks."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT track_id, GROUP_CONCAT(label, ',') as seq
               FROM (SELECT track_id, label FROM sections ORDER BY track_id, position)
               GROUP BY track_id""",
        ).fetchall()
        if not rows:
            return []
        sequences = [r[1].split(",") for r in rows if r[1]]
        return _most_common(sequences) or []
    finally:
        conn.close()


def _modal_real(conn, column: str) -> float | None:
    row = conn.execute(f"SELECT {column} FROM tracks WHERE {column} IS NOT NULL ORDER BY {column} LIMIT 1").fetchone()
    return row[0] if row else None


def _modal_text(conn, column: str) -> str | None:
    row = conn.execute(
        f"SELECT {column}, COUNT(*) as n FROM tracks WHERE {column} IS NOT NULL"
        f" GROUP BY {column} ORDER BY n DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def _most_common(sequences: list[list[str]]) -> list[str] | None:
    if not sequences:
        return None
    key = max(set(map(tuple, sequences)), key=lambda s: sequences.count(list(s)))
    return list(key)
