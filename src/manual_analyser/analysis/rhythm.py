"""
analysis/rhythm.py — Stage 3a: rhythm analysis.

Responsibilities:
  - Detect kick, snare, and hi-hat onset patterns from the drums stem
  - Encode each as a 16-step binary grid string (modal bar pattern)
  - Compute syncopation score and rhythmic density
  - Classify groove feel as "straight", "swung", or "unclear"
  - Write results to SQLite (beat_patterns table + tracks.groove_feel)

Writes to SQLite:
  INSERT INTO beat_patterns (track_id, kick_pattern, snare_pattern,
    hihat_pattern, syncopation_score, rhythmic_density)
  UPDATE tracks SET groove_feel = ? WHERE track_id = ?

Note: madmom is confirmed broken on Python 3.10+ (collections.MutableSequence
removed). This module uses librosa exclusively. See compatibility.md.

Error handling (per docs/ERROR_HANDLING.md):
  - Numerical errors → write null/default values, log warning, continue
  - Unhandled exception → write null fields, log error, continue
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from manual_analyser.db import get_connection
from manual_analyser.utils import (
    classify_groove_feel,
    normalise_rhythmic_density,
    onsets_to_pattern,
)

logger = logging.getLogger(__name__)

# Frequency band boundaries for kick/snare/hihat separation
KICK_FMAX = 200  # Hz — low end captures kick drum fundamentals
SNARE_FMIN = 200  # Hz
SNARE_FMAX = 3000  # Hz — snare body and crack
HIHAT_FMIN = 4000  # Hz — hi-hat and cymbal presence

# Number of steps in the beat grid pattern
PATTERN_STEPS = 16

# Null pattern — used when detection fails
NULL_PATTERN = "0" * PATTERN_STEPS


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RhythmResult:
    """Rhythm analysis results for a single track."""

    kick_pattern: str  # 16-char binary string
    snare_pattern: str  # 16-char binary string
    hihat_pattern: str  # 16-char binary string
    syncopation_score: float  # 0.0–1.0
    rhythmic_density: float  # 0.0–1.0 (normalised)
    groove_feel: str  # "straight" | "swung" | "unclear"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse_rhythm(
    track_id: str,
    drums_wav: Path,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> RhythmResult | None:
    """
    Analyse rhythm features from the drums stem and write to SQLite.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        drums_wav: Path to the drums stem WAV (output of separate stage).
        data_dir: Root data directory (default: "data/").
        db_path: Path to SQLite database. Defaults to data/manual_analyser.db.

    Returns:
        RhythmResult on success, or None if analysis failed.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        result = _compute_rhythm(drums_wav, short_id)
    except Exception as e:
        logger.error("[%s] [rhythm] Analysis failed: %s", short_id, e, exc_info=True)
        _write_nulls(resolved_db, track_id, short_id)
        return None

    _write_result(resolved_db, track_id, result, short_id)
    logger.info(
        "[%s] [rhythm] feel=%s sync=%.2f density=%.2f kick=%s snare=%s",
        short_id,
        result.groove_feel,
        result.syncopation_score,
        result.rhythmic_density,
        result.kick_pattern,
        result.snare_pattern,
    )
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _compute_rhythm(drums_wav: Path, short_id: str) -> RhythmResult:
    """
    Load the drums stem and compute rhythm features.

    Separates the drums stem into frequency bands to isolate kick, snare,
    and hi-hat, then detects onsets in each band and quantises them to a
    16-step grid. The modal pattern across all bars is returned.

    Args:
        drums_wav: Path to the drums stem WAV.
        short_id: First 8 chars of track_id for log messages.

    Returns:
        RhythmResult with all fields populated.
    """
    y, sr = librosa.load(str(drums_wav), sr=None, mono=True)
    logger.debug("[%s] [rhythm] Loaded drums stem %.1fs", short_id, len(y) / sr)

    # Get beat grid from the full drum signal
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_frames) < 4:
        logger.warning("[%s] [rhythm] Too few beats detected (%d)", short_id, len(beat_frames))
        return _null_result()

    # Onset detection per frequency band
    kick_onsets = _detect_onsets_in_band(y, sr, fmin=None, fmax=KICK_FMAX)
    snare_onsets = _detect_onsets_in_band(y, sr, fmin=SNARE_FMIN, fmax=SNARE_FMAX)
    hihat_onsets = _detect_onsets_in_band(y, sr, fmin=HIHAT_FMIN, fmax=None)

    # Quantise to 16-step patterns
    kick_pattern = onsets_to_pattern(kick_onsets, beat_frames, steps=PATTERN_STEPS)
    snare_pattern = onsets_to_pattern(snare_onsets, beat_frames, steps=PATTERN_STEPS)
    hihat_pattern = onsets_to_pattern(hihat_onsets, beat_frames, steps=PATTERN_STEPS)

    # Syncopation: ratio of off-beat onsets to total onsets
    all_onsets = np.unique(np.concatenate([kick_onsets, snare_onsets, hihat_onsets]))
    syncopation = _compute_syncopation(all_onsets, beat_frames, sr)

    # Rhythmic density: average onsets per beat, normalised
    raw_density = len(all_onsets) / max(len(beat_frames), 1)
    rhythmic_density = normalise_rhythmic_density(raw_density)

    # Groove feel classification
    groove_feel = classify_groove_feel(beat_times, sr)

    return RhythmResult(
        kick_pattern=kick_pattern,
        snare_pattern=snare_pattern,
        hihat_pattern=hihat_pattern,
        syncopation_score=syncopation,
        rhythmic_density=rhythmic_density,
        groove_feel=groove_feel,
    )


def _detect_onsets_in_band(
    y: np.ndarray,
    sr: int,
    fmin: float | None,
    fmax: float | None,
) -> np.ndarray:
    """
    Detect onset frames in a specific frequency band.

    Applies a bandpass filter via the onset strength envelope, which
    internally uses a mel spectrogram bounded by fmin/fmax.

    Args:
        y: Audio signal (mono).
        sr: Sample rate.
        fmin: Lower frequency bound in Hz, or None for no lower limit.
        fmax: Upper frequency bound in Hz, or None for no upper limit.

    Returns:
        Array of onset frame indices.
    """
    kwargs: dict = {}
    if fmin is not None:
        kwargs["fmin"] = fmin
    if fmax is not None:
        kwargs["fmax"] = fmax

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, **kwargs)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units="frames",
            backtrack=True,
        )
        return onsets
    except Exception as e:
        logger.warning("Onset detection failed for band fmin=%s fmax=%s: %s", fmin, fmax, e)
        return np.array([], dtype=int)


def _compute_syncopation(
    onset_frames: np.ndarray,
    beat_frames: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> float:
    """
    Compute syncopation as the ratio of off-beat onsets to total onsets.

    An onset is "on-beat" if it falls within one hop of a beat frame.
    Everything else is "off-beat" (syncopated).

    Args:
        onset_frames: All onset frame indices.
        beat_frames: Beat frame indices.
        sr: Sample rate (unused directly, kept for interface consistency).
        hop_length: Hop length used in onset detection.

    Returns:
        Syncopation score 0.0–1.0. 0.0 = all on-beat, 1.0 = all off-beat.
    """
    if len(onset_frames) == 0 or len(beat_frames) == 0:
        return 0.0

    tolerance = 2  # frames — within 2 frames of a beat = "on beat"

    on_beat = 0
    for onset in onset_frames:
        distances = np.abs(beat_frames - onset)
        if distances.min() <= tolerance:
            on_beat += 1

    off_beat = len(onset_frames) - on_beat
    return float(np.clip(off_beat / len(onset_frames), 0.0, 1.0))


def _null_result() -> RhythmResult:
    """Return a RhythmResult with safe default values for failed analysis."""
    return RhythmResult(
        kick_pattern=NULL_PATTERN,
        snare_pattern=NULL_PATTERN,
        hihat_pattern=NULL_PATTERN,
        syncopation_score=0.0,
        rhythmic_density=0.0,
        groove_feel="unclear",
    )


# ---------------------------------------------------------------------------
# SQLite writes
# ---------------------------------------------------------------------------


def _write_result(
    db_path: Path,
    track_id: str,
    result: RhythmResult,
    short_id: str,
) -> None:
    """Write RhythmResult to beat_patterns table and tracks.groove_feel."""
    conn = get_connection(db_path)
    try:
        with conn:
            # Insert beat pattern row
            conn.execute(
                """
                INSERT INTO beat_patterns (
                    track_id, kick_pattern, snare_pattern, hihat_pattern,
                    syncopation_score, rhythmic_density
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    track_id,
                    result.kick_pattern,
                    result.snare_pattern,
                    result.hihat_pattern,
                    round(result.syncopation_score, 4),
                    round(result.rhythmic_density, 4),
                ),
            )
            # Update groove_feel on tracks row
            conn.execute(
                "UPDATE tracks SET groove_feel = ? WHERE track_id = ?",
                (result.groove_feel, track_id),
            )
    finally:
        conn.close()


def _write_nulls(db_path: Path, track_id: str, short_id: str) -> None:
    """Write safe defaults when analysis fails entirely."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO beat_patterns (
                    track_id, kick_pattern, snare_pattern, hihat_pattern,
                    syncopation_score, rhythmic_density
                ) VALUES (?, ?, ?, ?, NULL, NULL)
                """,
                (track_id, NULL_PATTERN, NULL_PATTERN, NULL_PATTERN),
            )
            conn.execute(
                "UPDATE tracks SET groove_feel = 'unclear' WHERE track_id = ?",
                (track_id,),
            )
        logger.warning("[%s] [rhythm] Wrote null fields due to analysis failure", short_id)
    finally:
        conn.close()
