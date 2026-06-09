"""
analysis/groove/danceability.py — Danceability computation.

Provides the essentia-based Danceability descriptor with a librosa
approximation fallback for when essentia is unavailable or fails.
"""

import logging

import librosa
import numpy as np

from manual_analyser.analysis.groove.regularity import _compute_beat_regularity

logger = logging.getLogger(__name__)


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

        y_float32 = y.astype(np.float32)

        if sr != 44100:
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

    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    if len(beat_frames) >= 3:
        ibis = np.diff(librosa.frames_to_time(beat_frames, sr=sr))
        mean_ibi = float(np.mean(ibis))
        cv = float(np.std(ibis) / mean_ibi) if mean_ibi > 0 else 1.0
        tempo_stability = float(np.clip(1.0 - cv / 0.2, 0.0, 1.0))
    else:
        tempo_stability = 0.5

    rms = float(librosa.feature.rms(y=y).mean())
    energy_factor = float(np.clip(rms * 10, 0.0, 1.0))

    return float(np.clip(beat_regularity * 0.5 + tempo_stability * 0.3 + energy_factor * 0.2, 0.0, 1.0))
