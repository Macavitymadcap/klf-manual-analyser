"""
analysis/harmony/sections.py — Section boundary helpers and SQLite writes.

Imports types from harmony.types (not from harmony.__init__) to avoid
circular imports.
"""

import json
import logging
from pathlib import Path

from manual_analyser.analysis.harmony.types import HarmonyResult
from manual_analyser.db import get_connection

logger = logging.getLogger(__name__)


def _get_section_boundaries(
    db_path: Path,
    track_id: str,
    duration: float,
    n_fallback: int = 8,
) -> list[tuple[float, float]]:
    """
    Get section boundaries from the database.

    If structure.py has already written sections, use those boundaries.
    Otherwise, divide the track into n_fallback equal-length segments.

    Args:
        db_path: SQLite path.
        track_id: Full track ID.
        duration: Track duration in seconds.
        n_fallback: Number of equal segments to use if no sections exist.

    Returns:
        List of (start, end) tuples in seconds.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT start, end FROM sections WHERE track_id = ? ORDER BY position",
            (track_id,),
        ).fetchall()
    finally:
        conn.close()

    if rows:
        return [(row["start"], row["end"]) for row in rows]

    segment_len = duration / n_fallback
    return [(i * segment_len, min((i + 1) * segment_len, duration)) for i in range(n_fallback)]


def _write_result(
    db_path: Path,
    track_id: str,
    result: HarmonyResult,
    short_id: str,
) -> None:
    """Write HarmonyResult to tracks, sections, and chord_progressions tables."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET key = ?, mode = ?, key_confidence = ?
                WHERE track_id = ?
                """,
                (result.key, result.mode, round(result.key_confidence, 4), track_id),
            )

            existing = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE track_id = ?",
                (track_id,),
            ).fetchone()[0]

            for section in result.sections:
                if existing == 0:
                    cursor = conn.execute(
                        """
                        INSERT INTO sections
                            (track_id, position, start, end, duration,
                             label, label_confidence, label_source)
                        VALUES (?, ?, ?, ?, ?, 'unknown', 0.0, 'acoustic')
                        """,
                        (
                            track_id,
                            section.position,
                            round(section.start, 3),
                            round(section.end, 3),
                            round(section.end - section.start, 3),
                        ),
                    )
                    section_id = cursor.lastrowid
                else:
                    row = conn.execute(
                        "SELECT id FROM sections WHERE track_id = ? AND position = ?",
                        (track_id, section.position),
                    ).fetchone()
                    section_id = row["id"] if row else None

                if section_id is None:
                    continue

                chords_json = json.dumps([{"start": c.start, "end": c.end, "chord": c.chord} for c in section.chords])
                conn.execute(
                    """
                    INSERT INTO chord_progressions (section_id, progression, chords_json)
                    VALUES (?, ?, ?)
                    """,
                    (section_id, section.progression, chords_json),
                )

    finally:
        conn.close()


def _write_nulls(db_path: Path, track_id: str, short_id: str) -> None:
    """Write null for harmony fields when analysis fails."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET key = NULL, mode = NULL, key_confidence = NULL
                WHERE track_id = ?
                """,
                (track_id,),
            )
        logger.warning("[%s] [harmony] Wrote null fields due to failure", short_id)
    finally:
        conn.close()
