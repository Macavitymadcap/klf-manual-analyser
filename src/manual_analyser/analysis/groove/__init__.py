"""
analysis/groove — Groove feature analysis for the KLF Manual Analyser.

Public API:
    analyse_groove(track_id, full_wav, data_dir, db_path) -> GrooveResult | None
    GrooveResult

Previously-private functions are re-exported here so that existing test
imports (e.g. `from manual_analyser.analysis.groove import _compute_beat_regularity`)
continue to work without change.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from manual_analyser.db import get_connection

from .danceability import _approximate_danceability, _compute_danceability
from .regularity import _compute_beat_regularity, _compute_repetition_score, _compute_self_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GrooveResult:
    """Groove analysis results for a single track."""

    danceability: float  # 0.0–1.0 (essentia or approximation)
    self_similarity_score: float  # 0.0–1.0
    beat_regularity: float  # 0.0–1.0
    groove_consistency: float  # composite: sqrt(beat_regularity * self_similarity)
    repetition_score: float  # 0.0–1.0; chroma-based


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse_groove(
    track_id: str,
    full_wav: Path,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> GrooveResult | None:
    """
    Compute groove features from the full mix WAV and write to SQLite.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        full_wav: Path to the decoded mono WAV.
        data_dir: Root data directory (default: "data/").
        db_path: Path to SQLite database. Defaults to data/manual_analyser.db.

    Returns:
        GrooveResult on success, or None if analysis failed.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        result = _compute_groove(full_wav, short_id)
    except Exception as e:
        logger.exception("[%s] [groove] Analysis failed: %s", short_id, e, exc_info=True)
        _write_nulls(resolved_db, track_id, short_id)
        return None

    _write_result(resolved_db, track_id, result, short_id)
    logger.info(
        "[%s] [groove] dance=%.2f sim=%.2f reg=%.2f consist=%.2f rep=%.2f",
        short_id,
        result.danceability,
        result.self_similarity_score,
        result.beat_regularity,
        result.groove_consistency,
        result.repetition_score,
    )
    return result


# ---------------------------------------------------------------------------
# Analysis orchestration
# ---------------------------------------------------------------------------


def _compute_groove(full_wav: Path, short_id: str) -> GrooveResult:
    """
    Load audio and compute all groove features.

    Args:
        full_wav: Path to the full mix WAV.
        short_id: First 8 chars of track_id for log messages.

    Returns:
        GrooveResult with all fields populated.
    """
    y, sr = librosa.load(str(full_wav), sr=None, mono=True)
    logger.debug("[%s] [groove] Loaded %.1fs at %dHz", short_id, len(y) / sr, sr)

    danceability = _compute_danceability(y, sr, short_id)
    beat_regularity = _compute_beat_regularity(y, sr)
    self_similarity = _compute_self_similarity(y, sr)
    repetition_score = _compute_repetition_score(y, sr)

    # Geometric mean penalises when either component is low
    groove_consistency = float(np.sqrt(beat_regularity * self_similarity))

    return GrooveResult(
        danceability=danceability,
        self_similarity_score=self_similarity,
        beat_regularity=beat_regularity,
        groove_consistency=groove_consistency,
        repetition_score=repetition_score,
    )


# ---------------------------------------------------------------------------
# SQLite writes
# ---------------------------------------------------------------------------


def _write_result(
    db_path: Path,
    track_id: str,
    result: GrooveResult,
    short_id: str,
) -> None:
    """Write GrooveResult fields to the tracks table."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET
                    danceability = ?,
                    self_similarity_score = ?,
                    beat_regularity = ?,
                    groove_consistency = ?,
                    repetition_score = ?
                WHERE track_id = ?
                """,
                (
                    round(result.danceability, 4),
                    round(result.self_similarity_score, 4),
                    round(result.beat_regularity, 4),
                    round(result.groove_consistency, 4),
                    round(result.repetition_score, 4),
                    track_id,
                ),
            )
    finally:
        conn.close()


def _write_nulls(db_path: Path, track_id: str, short_id: str) -> None:
    """Write null for all groove fields when analysis fails."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET
                    danceability = NULL,
                    self_similarity_score = NULL,
                    beat_regularity = NULL,
                    groove_consistency = NULL,
                    repetition_score = NULL
                WHERE track_id = ?
                """,
                (track_id,),
            )
        logger.warning("[%s] [groove] Wrote null fields due to analysis failure", short_id)
    finally:
        conn.close()


# Re-export private helpers so existing test imports don't break.
# Tests import these directly: from manual_analyser.analysis.groove import _compute_beat_regularity
__all__ = [
    "GrooveResult",
    "analyse_groove",
    "_compute_groove",
    "_compute_danceability",
    "_approximate_danceability",
    "_compute_beat_regularity",
    "_compute_self_similarity",
    "_compute_repetition_score",
    "_write_result",
    "_write_nulls",
]
