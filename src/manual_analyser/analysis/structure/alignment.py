"""
analysis/structure/alignment.py — Stage 4: hybrid section alignment (pass 2).

Reads section boundaries, RMS energy profile, and transcript timestamps
from SQLite, then delegates label assignment to labelling.py, and writes
the results back.

This module owns DB I/O and pipeline orchestration. The label heuristics
themselves live in labelling.py.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from manual_analyser.analysis.normalise import normalise_lyric_density
from manual_analyser.analysis.structure.labelling import _assign_labels
from manual_analyser.analysis.structure.types import SectionLabel
from manual_analyser.db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_sections(
    track_id: str,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> list[SectionLabel] | None:
    """
    Assign section labels by cross-referencing acoustic and lyric signals.

    Reads from SQLite:
    - Section boundaries (from pass 1 or harmony.py)
    - RMS energy profile (from energy.py)
    - Transcript segments (from whisper.py)

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        data_dir: Root data directory.
        db_path: Path to SQLite database.

    Returns:
        List of SectionLabel objects, or None if alignment failed.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        labels = _run_alignment(resolved_db, track_id, short_id)
    except Exception as e:
        logger.exception("[%s] [structure/pass2] Alignment failed: %s", short_id, e, exc_info=True)
        return None

    if labels:
        _write_labels(resolved_db, track_id, labels, short_id)
        logger.info(
            "[%s] [structure/pass2] Labelled %d sections: %s",
            short_id,
            len(labels),
            [label.label for label in labels],
        )

    return labels


# ---------------------------------------------------------------------------
# Internal orchestration
# ---------------------------------------------------------------------------


def _run_alignment(
    db_path: Path,
    track_id: str,
    short_id: str,
) -> list[SectionLabel]:
    """
    Load all data from the DB and run the labelling algorithm.

    Args:
        db_path: SQLite path.
        track_id: Full track ID.
        short_id: For log messages.

    Returns:
        List of SectionLabel objects.
    """
    conn = get_connection(db_path)
    try:
        section_rows = conn.execute(
            "SELECT id, position, start, end FROM sections WHERE track_id = ? ORDER BY position",
            (track_id,),
        ).fetchall()

        if not section_rows:
            logger.warning("[%s] [structure/pass2] No sections found", short_id)
            return []

        ts_row = conn.execute(
            "SELECT rms_profile_json FROM tracks_timeseries WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        rms_profile = json.loads(ts_row["rms_profile_json"]) if ts_row else []

        transcript_rows = conn.execute(
            "SELECT start, end, text FROM transcript_segments WHERE track_id = ? ORDER BY start",
            (track_id,),
        ).fetchall()

        track_row = conn.execute(
            "SELECT duration FROM tracks WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        duration = float(track_row["duration"]) if track_row else 0.0

    finally:
        conn.close()

    sections = [{"id": r["id"], "pos": r["position"], "start": r["start"], "end": r["end"]} for r in section_rows]

    rms_array = np.array(rms_profile) if rms_profile else np.array([])
    energies = _compute_section_energies(sections, rms_array, duration)
    lyric_data = _compute_lyric_features(sections, transcript_rows)

    return _assign_labels(sections, energies, lyric_data, duration, short_id)


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------


def _compute_section_energies(
    sections: list[dict],
    rms_array: np.ndarray,
    duration: float,
    rms_interval: float = 0.5,
) -> list[float]:
    """
    Compute mean normalised RMS energy for each section.

    Args:
        sections: List of section dicts with start/end.
        rms_array: Normalised RMS profile (one value per rms_interval seconds).
        duration: Total track duration in seconds.
        rms_interval: Sampling interval of RMS profile.

    Returns:
        List of mean energy values (0.0–1.0) per section.
    """
    if len(rms_array) == 0:
        return [0.5] * len(sections)

    energies = []
    for sec in sections:
        start_idx = int(sec["start"] / rms_interval)
        end_idx = int(sec["end"] / rms_interval) + 1
        start_idx = max(0, min(start_idx, len(rms_array) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(rms_array)))
        section_rms = rms_array[start_idx:end_idx]
        energies.append(float(np.mean(section_rms)) if len(section_rms) > 0 else 0.5)

    return energies


def _compute_lyric_features(
    sections: list[dict],
    transcript_rows: list,
) -> list[dict]:
    """
    Compute lyric density and repeated phrase for each section.

    Args:
        sections: List of section dicts with start/end.
        transcript_rows: Rows from transcript_segments table.

    Returns:
        List of dicts with 'lyric_density', 'word_count', 'phrases',
        'repeated_phrase' per section.
    """
    results = []

    for sec in sections:
        duration = sec["end"] - sec["start"]
        words_in_section = []
        for row in transcript_rows:
            if row["end"] > sec["start"] and row["start"] < sec["end"]:
                words_in_section.extend(row["text"].strip().split())

        word_count = len(words_in_section)
        raw_density = word_count / max(duration, 1.0)
        lyric_density = normalise_lyric_density(raw_density)
        repeated_phrase = _find_repeated_phrase(words_in_section)

        results.append(
            {
                "lyric_density": lyric_density,
                "word_count": word_count,
                "phrases": _extract_phrases(words_in_section),
                "repeated_phrase": repeated_phrase,
            }
        )

    return results


def _extract_phrases(words: list[str], n: int = 3) -> Counter:
    """Extract n-gram phrase counts from a word list."""
    if len(words) < n:
        return Counter()
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return Counter(ngrams)


def _find_repeated_phrase(words: list[str]) -> str | None:
    """Find the most repeated 3-word phrase, or None if no repetition."""
    phrases = _extract_phrases(words, n=3)
    if not phrases:
        return None
    most_common = phrases.most_common(1)[0]
    return most_common[0] if most_common[1] >= 2 else None


# ---------------------------------------------------------------------------
# SQLite writes
# ---------------------------------------------------------------------------


def _write_labels(
    db_path: Path,
    track_id: str,
    labels: list[SectionLabel],
    short_id: str,
) -> None:
    """Update section rows with alignment results."""
    conn = get_connection(db_path)
    try:
        with conn:
            for section in labels:
                conn.execute(
                    """
                    UPDATE sections SET
                        label = ?,
                        label_confidence = ?,
                        label_source = ?,
                        mean_energy = ?,
                        lyric_density = ?,
                        repeated_phrase = ?
                    WHERE track_id = ? AND position = ?
                    """,
                    (
                        section.label,
                        round(section.label_confidence, 4),
                        section.label_source,
                        round(section.mean_energy, 4),
                        round(section.lyric_density, 4),
                        section.repeated_phrase,
                        track_id,
                        section.position,
                    ),
                )
    finally:
        conn.close()
