"""embedding/db_reader.py — Read track features from SQLite for summarisation."""

from dataclasses import dataclass
from pathlib import Path

from manual_analyser.db import get_connection


@dataclass
class TrackFeatures:
    """All fields needed to build a feature summary."""

    track_id: str
    artist: str | None
    song_name: str | None
    bpm: float | None
    key: str | None
    mode: str | None
    groove_feel: str | None
    energy_shape: str | None
    danceability: float | None
    hook_phrase: str | None
    hook_repetition_count: int | None
    unique_word_ratio: float | None
    section_labels: list[str]
    kick_pattern: str | None
    snare_pattern: str | None


def load_track_features(track_id: str, db_path: Path) -> TrackFeatures | None:
    """Load all features needed for embedding from SQLite. Returns None if track missing."""
    conn = get_connection(db_path)
    try:
        row = _fetch_track_row(conn, track_id)
        if row is None:
            return None
        sections = _fetch_section_labels(conn, track_id)
        patterns = _fetch_beat_patterns(conn, track_id)
        return _build_features(track_id, row, sections, patterns)
    finally:
        conn.close()


def _fetch_track_row(conn, track_id: str) -> dict | None:
    row = conn.execute(
        """SELECT artist, song_name, bpm, key, mode, groove_feel,
                  energy_shape, danceability, hook_phrase,
                  hook_repetition_count, unique_word_ratio
           FROM tracks WHERE track_id = ?""",
        (track_id,),
    ).fetchone()
    return dict(row) if row else None


def _fetch_section_labels(conn, track_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT label FROM sections WHERE track_id = ? ORDER BY position",
        (track_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _fetch_beat_patterns(conn, track_id: str) -> dict | None:
    row = conn.execute(
        "SELECT kick_pattern, snare_pattern FROM beat_patterns WHERE track_id = ?",
        (track_id,),
    ).fetchone()
    return dict(row) if row else None


def _build_features(track_id: str, row: dict, sections: list[str], patterns: dict | None) -> TrackFeatures:
    return TrackFeatures(
        track_id=track_id,
        artist=row["artist"],
        song_name=row["song_name"],
        bpm=row["bpm"],
        key=row["key"],
        mode=row["mode"],
        groove_feel=row["groove_feel"],
        energy_shape=row["energy_shape"],
        danceability=row["danceability"],
        hook_phrase=row["hook_phrase"],
        hook_repetition_count=row["hook_repetition_count"],
        unique_word_ratio=row["unique_word_ratio"],
        section_labels=sections,
        kick_pattern=patterns["kick_pattern"] if patterns else None,
        snare_pattern=patterns["snare_pattern"] if patterns else None,
    )
