import logging
from pathlib import Path

import librosa
import numpy as np

from manual_analyser.analysis.energy.types import RMS_SAMPLE_INTERVAL, EnergyResult
from manual_analyser.analysis.normalise import normalise_dynamic_range, normalise_loudness, normalise_verse_chorus_delta
from manual_analyser.db import get_connection

logger = logging.getLogger(__name__)


def _compute_energy(
    full_wav: Path,
    short_id: str,
    db_path: Path,
    track_id: str,
) -> EnergyResult:
    """
    Load audio and compute all energy features.

    Args:
        full_wav: Path to the full mix WAV.
        short_id: First 8 chars of track_id for log messages.
        db_path: SQLite path (used to read section labels if available).
        track_id: Full track ID.

    Returns:
        EnergyResult with all fields populated.
    """
    y, sr = librosa.load(str(full_wav), sr=None, mono=True)
    logger.debug("[%s] [energy] Loaded %.1fs at %dHz", short_id, len(y) / sr, sr)

    # RMS profile — sampled every RMS_SAMPLE_INTERVAL seconds
    hop_samples = int(sr * RMS_SAMPLE_INTERVAL)
    rms_frames = librosa.feature.rms(y=y, frame_length=hop_samples * 2, hop_length=hop_samples)[0]

    # Normalise RMS to 0–1 range
    rms_max = float(rms_frames.max())
    if rms_max > 0:
        rms_normalised = (rms_frames / rms_max).tolist()
    else:
        rms_normalised = [0.0] * len(rms_frames)

    # Integrated loudness (LUFS)
    loudness_db = _compute_loudness(y, sr)

    # Dynamic range
    dynamic_range_db = _compute_dynamic_range(y, sr)

    # Energy shape from the RMS profile
    energy_shape = _classify_energy_shape(np.array(rms_normalised))

    # Verse/chorus delta — use section labels if available, else half-split
    verse_chorus_delta = _compute_verse_chorus_delta(
        np.array(rms_normalised), sr, hop_samples, db_path, track_id, short_id
    )

    return EnergyResult(
        loudness_db=loudness_db,
        dynamic_range_db=dynamic_range_db,
        verse_chorus_delta=verse_chorus_delta,
        energy_shape=energy_shape,
        rms_profile=rms_normalised,
    )


def _compute_loudness(y: np.ndarray, sr: int) -> float:
    """
    Compute integrated loudness in LUFS using pyloudnorm and normalise.

    Falls back to RMS-based approximation if pyloudnorm fails.

    Args:
        y: Audio signal.
        sr: Sample rate.

    Returns:
        Normalised loudness 0.0–1.0.
    """
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        # pyloudnorm returns -inf for silence; clamp to -60
        if not np.isfinite(loudness):
            loudness = -60.0
        return normalise_loudness(float(loudness))
    except Exception as e:
        logger.warning("pyloudnorm failed (%s), falling back to RMS", e)
        rms = float(np.sqrt(np.mean(y**2)))
        if rms == 0:
            return 0.0
        lufs_approx = 20 * np.log10(rms) - 0.691
        return normalise_loudness(float(lufs_approx))


def _compute_dynamic_range(y: np.ndarray, sr: int) -> float:
    """
    Compute dynamic range as the difference between peak and floor RMS
    across short analysis frames, normalised to 0.0–1.0.

    Args:
        y: Audio signal.
        sr: Sample rate.

    Returns:
        Normalised dynamic range 0.0–1.0.
    """
    frame_length = sr // 4  # 250ms frames
    hop_length = frame_length // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    rms_db = librosa.amplitude_to_db(rms + 1e-10)  # avoid log(0)

    # Dynamic range = difference between 95th and 5th percentile
    # (avoids noise floor and clipping artefacts at extremes)
    p95 = float(np.percentile(rms_db, 95))
    p5 = float(np.percentile(rms_db, 5))
    raw_range = max(0.0, p95 - p5)

    return normalise_dynamic_range(raw_range)


def _classify_energy_shape(rms: np.ndarray) -> str:
    """
    Classify the overall energy trajectory of the track.

    Fits a linear trend to the RMS profile and uses correlation with
    quadratic and linear models to classify:

    - "building": energy increases through the track (positive linear trend)
    - "peaked": energy peaks in the middle (inverted U shape)
    - "flat": consistent energy throughout (low variance)
    - "unclear": does not fit any pattern clearly

    Args:
        rms: Normalised RMS profile array.

    Returns:
        "building", "flat", "peaked", or "unclear".
    """
    if len(rms) < 4:
        return "unclear"

    n = len(rms)
    x = np.linspace(0, 1, n)

    # Linear fit
    linear_coeffs = np.polyfit(x, rms, 1)
    linear_pred = np.polyval(linear_coeffs, x)
    linear_corr = float(np.corrcoef(rms, linear_pred)[0, 1])

    # Quadratic fit (for peaked shape)
    quad_coeffs = np.polyfit(x, rms, 2)
    quad_pred = np.polyval(quad_coeffs, x)
    quad_corr = float(np.corrcoef(rms, quad_pred)[0, 1])

    # Variance check for flat
    rms_std = float(np.std(rms))

    # Classification thresholds
    if rms_std < 0.08:
        return "flat"

    if quad_corr > 0.7 and quad_coeffs[0] < -0.1:
        # Strong inverted U — peaked
        return "peaked"

    if linear_corr > 0.6 and linear_coeffs[0] > 0.05:
        # Strong positive slope — building
        return "building"

    return "unclear"


def _compute_verse_chorus_delta(
    rms: np.ndarray,
    sr: int,
    hop_samples: int,
    db_path: Path,
    track_id: str,
    short_id: str,
) -> float:
    """
    Estimate the energy lift between verse and chorus sections.

    If labelled sections exist in the DB, uses mean RMS of all verse
    sections vs mean RMS of all chorus sections.

    If no sections exist yet, falls back to comparing the energy of the
    first quarter of the track (proxy for verse) against the third quarter
    (proxy for chorus peak).

    Args:
        rms: Normalised RMS profile (one value per hop_samples).
        sr: Sample rate.
        hop_samples: Samples per RMS frame.
        db_path: SQLite path.
        track_id: Full track ID.
        short_id: First 8 chars for logging.

    Returns:
        Normalised delta 0.0–1.0.
    """
    # Try section-based calculation first
    try:
        delta = _delta_from_sections(rms, sr, hop_samples, db_path, track_id)
        if delta is not None:
            return delta
    except Exception as e:
        logger.debug("[%s] [energy] Section-based delta failed: %s", short_id, e)

    # Fallback: first quarter vs third quarter comparison
    return _delta_from_halves(rms)


def _delta_from_sections(
    rms: np.ndarray,
    sr: int,
    hop_samples: int,
    db_path: Path,
    track_id: str,
) -> float | None:
    """
    Compute verse/chorus energy delta using section labels from SQLite.

    Returns None if no usable sections exist.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT label, start, end FROM sections
            WHERE track_id = ? AND label IN ('verse', 'chorus')
            ORDER BY start
            """,
            (track_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    verse_energies = []
    chorus_energies = []

    for row in rows:
        # Convert time range to RMS frame indices
        start_frame = int(row["start"] * sr / hop_samples)
        end_frame = int(row["end"] * sr / hop_samples)
        start_frame = max(0, min(start_frame, len(rms) - 1))
        end_frame = max(start_frame + 1, min(end_frame, len(rms)))

        section_rms = rms[start_frame:end_frame]
        if len(section_rms) == 0:
            continue

        mean_energy = float(np.mean(section_rms))
        if row["label"] == "verse":
            verse_energies.append(mean_energy)
        else:
            chorus_energies.append(mean_energy)

    if not verse_energies or not chorus_energies:
        return None

    verse_mean = float(np.mean(verse_energies))
    chorus_mean = float(np.mean(chorus_energies))

    # Convert RMS ratio to approximate dB difference
    if verse_mean == 0:
        return 0.0
    delta_db = max(0.0, 20 * np.log10(chorus_mean / (verse_mean + 1e-10)))
    return normalise_verse_chorus_delta(delta_db)


def _delta_from_halves(rms: np.ndarray) -> float:
    """
    Fallback: compare energy in first quarter vs third quarter of track.

    The first quarter is a rough proxy for verse material; the third
    quarter often contains a chorus or peak energy section.

    Args:
        rms: Normalised RMS profile.

    Returns:
        Normalised delta 0.0–1.0.
    """
    if len(rms) < 4:
        return 0.0

    n = len(rms)
    q1 = rms[: n // 4]
    q3 = rms[n // 2 : 3 * n // 4]

    verse_mean = float(np.mean(q1)) if len(q1) > 0 else 0.0
    chorus_mean = float(np.mean(q3)) if len(q3) > 0 else 0.0

    if verse_mean == 0:
        return 0.0
    delta_db = max(0.0, 20 * np.log10(chorus_mean / (verse_mean + 1e-10)))
    return normalise_verse_chorus_delta(delta_db)
