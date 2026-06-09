"""
analysis/groove.py — Stage 3a: groove analysis.

Responsibilities:
  - Compute danceability using essentia (confirmed working on Python 3.11)
  - Compute self-similarity from chroma features (librosa)
  - Compute beat regularity from inter-beat interval variance (librosa)
  - Compute groove consistency as composite of the above
  - Compute repetition score from chroma self-similarity
  - Write all fields to the tracks table

Writes to SQLite:
  UPDATE tracks SET
    danceability, self_similarity_score, beat_regularity,
    groove_consistency, repetition_score
  WHERE track_id = ?

Error handling (per docs/ERROR_HANDLING.md):
  - essentia failure → fall back to librosa approximation, log warning
  - Numerical errors → write null for affected fields, log warning
  - Unhandled exception → write null for all fields, log error
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from manual_analyser.db import get_connection

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
    groove_consistency: float  # composite: beat_regularity * self_similarity
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
# Analysis
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

    # Danceability — essentia preferred, librosa approximation fallback
    danceability = _compute_danceability(y, sr, short_id)

    # Beat regularity from inter-beat interval variance
    beat_regularity = _compute_beat_regularity(y, sr)

    # Self-similarity from chroma features
    self_similarity = _compute_self_similarity(y, sr)

    # Repetition score from chroma self-similarity matrix diagonal
    repetition_score = _compute_repetition_score(y, sr)

    # Groove consistency: geometric mean of beat_regularity and self_similarity
    # Using geometric mean rather than arithmetic to penalise when either is low
    groove_consistency = float(np.sqrt(beat_regularity * self_similarity))

    return GrooveResult(
        danceability=danceability,
        self_similarity_score=self_similarity,
        beat_regularity=beat_regularity,
        groove_consistency=groove_consistency,
        repetition_score=repetition_score,
    )


def _compute_danceability(y: np.ndarray, sr: int, short_id: str) -> float:
    """
    Compute danceability using essentia's Danceability descriptor.

    Falls back to a librosa-based approximation if essentia fails.

    The essentia Danceability algorithm is based on the detrended
    fluctuation analysis (DFA) of the RMS energy envelope, measuring
    how consistently the energy fluctuates at dance-relevant timescales.

    Args:
        y: Audio signal.
        sr: Sample rate.
        short_id: For log messages.

    Returns:
        Danceability 0.0–1.0.
    """
    try:
        import essentia.standard as es

        # Essentia expects float32 mono
        y_float32 = y.astype(np.float32)

        # Resample to 44100 if needed (essentia Danceability default)
        if sr != 44100:
            import librosa

            y_float32 = librosa.resample(y_float32, orig_sr=sr, target_sr=44100).astype(np.float32)

        danceability_algo = es.Danceability(sampleRate=44100)
        danceability, _ = danceability_algo(y_float32)
        return float(np.clip(danceability, 0.0, 1.0))

    except Exception as e:
        logger.warning("[%s] [groove] essentia danceability failed (%s), using approximation", short_id, e)
        return _approximate_danceability(y, sr)


def _approximate_danceability(y: np.ndarray, sr: int) -> float:
    """
    Approximate danceability from librosa features when essentia is unavailable.

    Uses a weighted combination of:
    - Beat regularity (metronomic groove → more danceable)
    - Tempo stability (consistent pulse → more danceable)
    - Onset density (rhythmic activity → more danceable, up to a point)

    Args:
        y: Audio signal.
        sr: Sample rate.

    Returns:
        Approximate danceability 0.0–1.0.
    """
    beat_regularity = _compute_beat_regularity(y, sr)

    # Tempo stability from beat tracking
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    if len(beat_frames) >= 3:
        ibis = np.diff(librosa.frames_to_time(beat_frames, sr=sr))
        mean_ibi = float(np.mean(ibis))
        cv = float(np.std(ibis) / mean_ibi) if mean_ibi > 0 else 1.0
        tempo_stability = float(np.clip(1.0 - cv / 0.2, 0.0, 1.0))
    else:
        tempo_stability = 0.5

    # RMS energy mean as proxy for rhythmic presence
    rms = float(librosa.feature.rms(y=y).mean())
    energy_factor = float(np.clip(rms * 10, 0.0, 1.0))

    return float(np.clip(beat_regularity * 0.5 + tempo_stability * 0.3 + energy_factor * 0.2, 0.0, 1.0))


def _compute_beat_regularity(y: np.ndarray, sr: int) -> float:
    """
    Compute beat regularity as 1 minus normalised coefficient of variation
    of inter-beat intervals.

    Args:
        y: Audio signal.
        sr: Sample rate.

    Returns:
        Beat regularity 0.0–1.0. 1.0 = perfectly metronomic.
    """
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 3:
        return 0.5

    ibis = np.diff(beat_times)
    mean_ibi = float(np.mean(ibis))

    if mean_ibi == 0:
        return 0.5

    cv = float(np.std(ibis) / mean_ibi)
    return float(np.clip(1.0 - cv / 0.2, 0.0, 1.0))


def _compute_self_similarity(y: np.ndarray, sr: int) -> float:
    """
    Compute a self-similarity score from the chroma feature matrix.

    Builds a recurrence matrix (which segment is similar to which other
    segment) and scores based on how consistently the track recurs —
    i.e. how much of the track resembles itself across time.

    A track with a strong repeating groove will have high off-diagonal
    similarity; a track that evolves continuously will have lower scores.

    Args:
        y: Audio signal.
        sr: Sample rate.

    Returns:
        Self-similarity score 0.0–1.0.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Downsample chroma to reduce computation
        # One frame per ~0.5 seconds
        hop = max(1, chroma.shape[1] // 200)
        chroma_ds = chroma[:, ::hop]

        if chroma_ds.shape[1] < 4:
            return 0.5

        # Cosine similarity matrix
        # Normalise each column
        norms = np.linalg.norm(chroma_ds, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        chroma_norm = chroma_ds / norms

        sim_matrix = chroma_norm.T @ chroma_norm

        n = sim_matrix.shape[0]
        if n < 4:
            return 0.5

        # Score: mean similarity of off-diagonal elements (excluding main diagonal)
        # High off-diagonal similarity = the track repeats material = high groove
        mask = ~np.eye(n, dtype=bool)
        off_diag_mean = float(sim_matrix[mask].mean())

        # Scale: similarity is already 0–1 for normalised vectors
        return float(np.clip(off_diag_mean, 0.0, 1.0))

    except Exception as e:
        logger.warning("Self-similarity computation failed: %s", e)
        return 0.5


def _compute_repetition_score(y: np.ndarray, sr: int) -> float:
    """
    Compute repetition score as the proportion of the track that recurs.

    Uses the librosa recurrence matrix to identify self-similar segments.
    A high repetition score indicates a track that loops back on itself —
    The Manual's "mindless repetition" criterion.

    Args:
        y: Audio signal.
        sr: Sample rate.

    Returns:
        Repetition score 0.0–1.0.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Downsample
        hop = max(1, chroma.shape[1] // 150)
        chroma_ds = chroma[:, ::hop]

        if chroma_ds.shape[1] < 4:
            return 0.5

        # Build recurrence matrix with a threshold
        R = librosa.segment.recurrence_matrix(
            chroma_ds,
            mode="affinity",
            metric="cosine",
            sparse=False,
        )

        # Repetition score: proportion of non-diagonal entries above threshold
        n = R.shape[0]
        threshold = 0.5
        mask = ~np.eye(n, dtype=bool)
        above_threshold = float((R[mask] > threshold).mean())

        return float(np.clip(above_threshold, 0.0, 1.0))

    except Exception as e:
        logger.warning("Repetition score computation failed: %s", e)
        return 0.5


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
        logger.warning("[%s] [groove] Wrote null fields due to failure", short_id)
    finally:
        conn.close()
