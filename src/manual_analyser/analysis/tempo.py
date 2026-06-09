"""
analysis/tempo.py — Stage 3a: tempo analysis.

Responsibilities:
  - Estimate BPM from the full mix WAV using librosa
  - Compute beat grid (timestamps of each beat)
  - Estimate time signature (3 or 4)
  - Compute tempo stability (how metronomic the groove is)
  - Write results to the tracks table in SQLite

Writes to SQLite:
  UPDATE tracks SET
    bpm, bpm_confidence, time_signature, tempo_stability
  WHERE track_id = ?

Error handling (per docs/ERROR_HANDLING.md):
  - Numerical errors → write null for affected fields, log warning, continue
  - Unhandled exception → write null for all fields, log error, continue
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
class TempoResult:
    """Tempo analysis results for a single track."""

    bpm: float  # beats per minute (physical unit)
    bpm_confidence: float  # 0.0–1.0
    time_signature: int  # 3 or 4
    tempo_stability: float  # 0.0–1.0; 1.0 = perfectly metronomic
    beat_times: np.ndarray  # timestamps of each beat in seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse_tempo(
    track_id: str,
    full_wav: Path,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> TempoResult | None:
    """
    Estimate tempo features from the full mix WAV and write to SQLite.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        full_wav: Path to the decoded mono WAV.
        data_dir: Root data directory (default: "data/").
        db_path: Path to SQLite database. Defaults to data/manual_analyser.db.

    Returns:
        TempoResult on success, or None if analysis failed (fields written
        as null to SQLite).
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        result = _compute_tempo(full_wav, short_id)
    except Exception as e:
        logger.exception("[%s] [tempo] Analysis failed: %s", short_id, e, exc_info=True)
        _write_nulls(resolved_db, track_id, short_id)
        return None

    _write_result(resolved_db, track_id, result, short_id)
    logger.info(
        "[%s] [tempo] BPM=%.1f confidence=%.2f time_sig=%d stability=%.2f",
        short_id,
        result.bpm,
        result.bpm_confidence,
        result.time_signature,
        result.tempo_stability,
    )
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _compute_tempo(full_wav: Path, short_id: str) -> TempoResult:
    """
    Load audio and compute tempo features using librosa.

    BPM estimation uses librosa.beat.beat_track with the onset envelope.
    Confidence is derived from the strength of the dominant tempo peak in
    the tempogram. Time signature is estimated from beat grouping patterns.
    Tempo stability is 1 minus the normalised standard deviation of
    inter-beat intervals.

    Args:
        full_wav: Path to mono WAV at 44100 Hz.
        short_id: First 8 chars of track_id for log messages.

    Returns:
        TempoResult with all fields populated.
    """
    # Load audio — librosa handles resampling if needed
    y, sr = librosa.load(str(full_wav), sr=None, mono=True)
    logger.debug("[%s] [tempo] Loaded %.1fs at %dHz", short_id, len(y) / sr, sr)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")

    # librosa 0.10+ returns tempo as a 1-element array
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # BPM confidence from tempogram
    bpm_confidence = _estimate_bpm_confidence(y, sr, bpm)

    # Time signature estimate
    time_sig = _estimate_time_signature(beat_times)

    # Tempo stability from inter-beat interval variance
    tempo_stability = _compute_tempo_stability(beat_times)

    return TempoResult(
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        time_signature=time_sig,
        tempo_stability=tempo_stability,
        beat_times=beat_times,
    )


def _estimate_bpm_confidence(y: np.ndarray, sr: int, bpm: float) -> float:
    """
    Estimate confidence in the BPM estimate from the tempogram.

    Computes the ratio of the dominant tempo peak energy to total tempogram
    energy. Higher ratio = clearer, more confident tempo.

    Args:
        y: Audio signal.
        sr: Sample rate.
        bpm: Estimated BPM.

    Returns:
        Confidence value 0.0–1.0.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

        # Sum energy across time for each tempo bin
        tempo_energy = tempogram.mean(axis=1)
        if tempo_energy.sum() == 0:
            return 0.5  # neutral fallback

        # Find the peak
        peak_idx = int(np.argmax(tempo_energy))
        peak_energy = float(tempo_energy[peak_idx])
        total_energy = float(tempo_energy.sum())

        confidence = float(np.clip(peak_energy / total_energy * 10, 0.0, 1.0))
        return confidence

    except Exception as e:
        logger.warning("BPM confidence estimation failed: %s", e)
        return 0.5


def _estimate_time_signature(beat_times: np.ndarray) -> int:
    """
    Estimate time signature (3 or 4) from beat grouping patterns.

    Uses autocorrelation of inter-beat intervals to detect whether beats
    group more naturally into 3s (waltz) or 4s (common time).

    Returns 4 by default — the vast majority of pop music is in 4/4 —
    and falls back to 4 if the analysis is inconclusive.

    Args:
        beat_times: Array of beat timestamps in seconds.

    Returns:
        3 or 4.
    """
    if len(beat_times) < 6:
        return 4  # not enough beats to determine

    ibis = np.diff(beat_times)
    if len(ibis) < 4:
        return 4

    # Compare autocorrelation at lag 3 vs lag 4
    # If lag-3 correlation is significantly stronger, likely 3/4
    try:
        ac = np.correlate(ibis - ibis.mean(), ibis - ibis.mean(), mode="full")
        ac = ac[len(ac) // 2 :]  # keep positive lags only

        if len(ac) < 5:
            return 4

        lag3 = float(ac[3]) if len(ac) > 3 else 0.0
        lag4 = float(ac[4]) if len(ac) > 4 else 0.0

        # Only classify as 3/4 if lag-3 is meaningfully stronger
        if lag3 > lag4 * 1.2:
            return 3
        return 4

    except Exception:
        return 4


def _compute_tempo_stability(beat_times: np.ndarray) -> float:
    """
    Compute tempo stability as 1 minus the normalised coefficient of
    variation of inter-beat intervals.

    A perfectly metronomic track (click track) returns 1.0.
    A track with highly variable timing returns close to 0.0.
    Pre-click-track recordings (1920s) will naturally score lower.

    Args:
        beat_times: Array of beat timestamps in seconds.

    Returns:
        Stability value 0.0–1.0.
    """
    if len(beat_times) < 3:
        return 0.5  # neutral fallback

    ibis = np.diff(beat_times)
    mean_ibi = float(np.mean(ibis))

    if mean_ibi == 0:
        return 0.5

    cv = float(np.std(ibis) / mean_ibi)  # coefficient of variation

    # CV of 0 = perfectly stable = 1.0
    # CV of 0.2 = quite variable = ~0.0 (clip)
    stability = float(np.clip(1.0 - (cv / 0.2), 0.0, 1.0))
    return stability


# ---------------------------------------------------------------------------
# SQLite writes
# ---------------------------------------------------------------------------


def _write_result(
    db_path: Path,
    track_id: str,
    result: TempoResult,
    short_id: str,
) -> None:
    """Write TempoResult fields to the tracks table."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET
                    bpm = ?,
                    bpm_confidence = ?,
                    time_signature = ?,
                    tempo_stability = ?
                WHERE track_id = ?
                """,
                (
                    round(result.bpm, 2),
                    round(result.bpm_confidence, 4),
                    result.time_signature,
                    round(result.tempo_stability, 4),
                    track_id,
                ),
            )
    finally:
        conn.close()


def _write_nulls(db_path: Path, track_id: str, short_id: str) -> None:
    """Write null for all tempo fields when analysis fails."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET
                    bpm = NULL,
                    bpm_confidence = NULL,
                    time_signature = NULL,
                    tempo_stability = NULL
                WHERE track_id = ?
                """,
                (track_id,),
            )
        logger.warning("[%s] [tempo] Wrote null fields due to analysis failure", short_id)
    finally:
        conn.close()
