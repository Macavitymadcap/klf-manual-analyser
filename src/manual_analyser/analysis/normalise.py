"""
analysis/normalise.py — Domain-specific normalisation functions for the analysis layer.

All analysis modules write fields to SQLite as normalised 0.0–1.0 values.
These functions convert raw physical measurements into that range.

Import:
    from manual_analyser.analysis.normalise import normalise_loudness, ...
"""

import numpy as np


def normalise_loudness(lufs: float) -> float:
    """
    Normalise integrated loudness from LUFS range [-60, 0] to [0.0, 1.0].

    Args:
        lufs: Integrated loudness in LUFS (typically -60 to 0).

    Returns:
        Normalised value clamped to [0.0, 1.0].
    """
    return float(np.clip((lufs + 60.0) / 60.0, 0.0, 1.0))


def normalise_dynamic_range(db: float) -> float:
    """
    Normalise dynamic range from [0, 60] dB to [0.0, 1.0].

    Args:
        db: Dynamic range in dB.

    Returns:
        Normalised value clamped to [0.0, 1.0].
    """
    return float(np.clip(db / 60.0, 0.0, 1.0))


def normalise_verse_chorus_delta(db: float) -> float:
    """
    Normalise verse-to-chorus energy delta from [0, 20] dB to [0.0, 1.0].

    The stored field is `verse_chorus_delta` (not `_db`).
    A normalised value of 0.15 corresponds to approximately 3dB lift.
    A normalised value of 0.30 corresponds to approximately 6dB lift.

    Args:
        db: Energy difference between chorus and verse in dB.

    Returns:
        Normalised value clamped to [0.0, 1.0].
    """
    return float(np.clip(db / 20.0, 0.0, 1.0))


def normalise_lyric_density(words_per_second: float) -> float:
    """
    Normalise lyric density from [0, ~5] words/sec to [0.0, 1.0].

    Args:
        words_per_second: Raw lyric density.

    Returns:
        Normalised value clamped to [0.0, 1.0].
    """
    return float(np.clip(words_per_second / 5.0, 0.0, 1.0))


def normalise_rhythmic_density(onsets_per_beat: float) -> float:
    """
    Normalise rhythmic density from [0, 4] onsets/beat to [0.0, 1.0].

    Args:
        onsets_per_beat: Raw rhythmic density.

    Returns:
        Normalised value clamped to [0.0, 1.0].
    """
    return float(np.clip(onsets_per_beat / 4.0, 0.0, 1.0))
