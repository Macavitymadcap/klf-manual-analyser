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

Uses librosa exclusively. madmom is confirmed broken on Python 3.10+.
See klf-mir-dev/references/compatibility.md.

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
        logger.exception("[%s] [rhythm] Analysis failed: %s", short_id, e, exc_info=True)
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

    Args:
        drums_wav: Path to the drums stem WAV.
        short_id: First 8 chars of track_id for log messages.

    Returns:
        RhythmResult with all fields populated.
    """
    y, sr = librosa.load(str(drums_wav), sr=None, mono=True)
    logger.debug("[%s] [rhythm] Loaded drums stem %.1fs", short_id, len(y) / sr)

    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_frames) < 4:
        logger.warning("[%s] [rhythm] Too few beats detected (%d)", short_id, len(beat_frames))
        return _null_result()

    kick_onsets = _detect_onsets_in_band(y, sr, fmin=None, fmax=KICK_FMAX)
    snare_onsets = _detect_onsets_in_band(y, sr, fmin=SNARE_FMIN, fmax=SNARE_FMAX)
    hihat_onsets = _detect_onsets_in_band(y, sr, fmin=HIHAT_FMIN, fmax=None)

    kick_pattern = onsets_to_pattern(kick_onsets, beat_frames, steps=PATTERN_STEPS)
    snare_pattern = onsets_to_pattern(snare_onsets, beat_frames, steps=PATTERN_STEPS)
    hihat_pattern = onsets_to_pattern(hihat_onsets, beat_frames, steps=PATTERN_STEPS)

    all_onsets = np.unique(np.concatenate([kick_onsets, snare_onsets, hihat_onsets]))
    syncopation = _compute_syncopation(all_onsets, beat_frames, sr)

    raw_density = len(all_onsets) / max(len(beat_frames), 1)
    rhythmic_density = normalise_rhythmic_density(raw_density)

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

    Args:
        onset_frames: All onset frame indices.
        beat_frames: Beat frame indices.
        sr: Sample rate (kept for interface consistency).
        hop_length: Hop length used in onset detection.

    Returns:
        Syncopation score 0.0–1.0. 0.0 = all on-beat, 1.0 = all off-beat.
    """
    if len(onset_frames) == 0 or len(beat_frames) == 0:
        return 0.0

    tolerance = 2  # frames

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
# Beat pattern encoding (moved from utils.py)
# ---------------------------------------------------------------------------


def onsets_to_pattern(
    onset_frames: np.ndarray,
    beat_frames: np.ndarray,
    steps: int = 16,
) -> str:
    """
    Quantise onset events to a 16-step binary grid string.

    Maps each onset to the nearest step in a single bar grid derived
    from the median inter-beat interval. Returns the modal pattern
    across all bars in the track.

    Args:
        onset_frames: Array of onset frame indices.
        beat_frames: Array of beat frame indices.
        steps: Number of steps per bar (default 16 = semiquaver resolution).

    Returns:
        16-character binary string, e.g. "1000100010001000" for four-on-the-floor.
        Returns "0" * steps if onset_frames or beat_frames is empty.
    """
    if len(onset_frames) == 0 or len(beat_frames) < 2:
        return "0" * steps

    ibi = float(np.median(np.diff(beat_frames)))
    step_size = ibi / (steps / 4)  # one beat = 4 sixteenth notes

    bar_patterns: list[list[int]] = []
    bar_length = ibi * 4
    start = beat_frames[0]
    n_bars = max(1, int((beat_frames[-1] - start) / bar_length))

    for bar_idx in range(n_bars):
        bar_start = start + bar_idx * bar_length
        bar_end = bar_start + bar_length
        pattern = [0] * steps

        bar_onsets = onset_frames[(onset_frames >= bar_start) & (onset_frames < bar_end)]

        for onset in bar_onsets:
            step = int((onset - bar_start) / step_size)
            if 0 <= step < steps:
                pattern[step] = 1

        bar_patterns.append(pattern)

    if not bar_patterns:
        return "0" * steps

    patterns_array = np.array(bar_patterns)
    modal = (patterns_array.mean(axis=0) >= 0.5).astype(int)
    return "".join(str(v) for v in modal)


# ---------------------------------------------------------------------------
# Groove feel classification (moved from utils.py)
# ---------------------------------------------------------------------------


def classify_groove_feel(
    beat_times: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> str:
    """
    Classify the rhythmic feel of a track as 'straight', 'swung', or 'unclear'.

    Uses the swing ratio: the ratio of the duration of the first 8th-note
    subdivision to the second within each beat. A perfectly straight feel
    has a ratio of 0.5 (equal division). Full triplet swing is 0.67.

    Thresholds:
        > 0.55  → 'swung'
        < 0.52  → 'straight'
        otherwise → 'unclear'

    Args:
        beat_times: Array of beat timestamps in seconds.
        sr: Sample rate.
        hop_length: Hop length used during beat tracking.

    Returns:
        'straight', 'swung', or 'unclear'.
    """
    if len(beat_times) < 8:
        return "unclear"

    swing_ratios = []
    for i in range(0, len(beat_times) - 2, 2):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        midpoint = beat_start + (beat_end - beat_start) / 2.0

        # Approximate: use ibi proportions as proxy for 8th-note timing
        first_half = midpoint - beat_start
        second_half = beat_end - midpoint
        total = first_half + second_half

        if total > 0:
            ratio = first_half / total
            swing_ratios.append(ratio)

    if not swing_ratios:
        return "unclear"

    mean_ratio = float(np.mean(swing_ratios))

    if mean_ratio > 0.55:
        return "swung"
    elif mean_ratio < 0.52:
        return "straight"
    else:
        return "unclear"


# ---------------------------------------------------------------------------
# Normalisation (moved from utils.py)
# ---------------------------------------------------------------------------


def normalise_rhythmic_density(onsets_per_beat: float) -> float:
    """
    Normalise rhythmic density from [0, 4] onsets/beat to [0.0, 1.0].

    Args:
        onsets_per_beat: Raw rhythmic density.

    Returns:
        Normalised value clamped to [0.0, 1.0].
    """
    return float(np.clip(onsets_per_beat / 4.0, 0.0, 1.0))


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
