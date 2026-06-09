"""
analysis/groove/regularity.py — Groove regularity and repetition metrics.

Provides the three librosa-based groove descriptors:
- Beat regularity from inter-beat interval variance
- Self-similarity from chroma recurrence matrix
- Repetition score from chroma recurrence matrix diagonal proportion
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


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

    Builds a recurrence matrix and scores based on how consistently the
    track recurs across time. A track with a strong repeating groove will
    have high off-diagonal similarity.

    Args:
        y: Audio signal.
        sr: Sample rate.

    Returns:
        Self-similarity score 0.0–1.0.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        hop = max(1, chroma.shape[1] // 200)
        chroma_ds = chroma[:, ::hop]

        if chroma_ds.shape[1] < 4:
            return 0.5

        norms = np.linalg.norm(chroma_ds, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        chroma_norm = chroma_ds / norms

        sim_matrix = chroma_norm.T @ chroma_norm

        n = sim_matrix.shape[0]
        if n < 4:
            return 0.5

        mask = ~np.eye(n, dtype=bool)
        off_diag_mean = float(sim_matrix[mask].mean())

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

        hop = max(1, chroma.shape[1] // 150)
        chroma_ds = chroma[:, ::hop]

        if chroma_ds.shape[1] < 4:
            return 0.5

        R = librosa.segment.recurrence_matrix(
            chroma_ds,
            mode="affinity",
            metric="cosine",
            sparse=False,
        )

        n = R.shape[0]
        threshold = 0.5
        mask = ~np.eye(n, dtype=bool)
        above_threshold = float((R[mask] > threshold).mean())

        return float(np.clip(above_threshold, 0.0, 1.0))

    except Exception as e:
        logger.warning("Repetition score computation failed: %s", e)
        return 0.5
