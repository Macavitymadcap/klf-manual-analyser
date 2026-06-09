# Chromatic scale note names
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (major and minor)
# Used to estimate key from chroma distribution
_MAJOR_PROFILE = np.array(
    [
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.52,
        5.19,
        2.39,
        3.66,
        2.29,
        2.88,
    ]
)
_MINOR_PROFILE = np.array(
    [
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17,
    ]
)


def _detect_key(chroma: np.ndarray) -> tuple[str, str, float]:
    """
    Detect musical key and mode using Krumhansl-Schmuckler key profiles.

    Correlates the mean chroma distribution against all 24 key profiles
    (12 major + 12 minor) and returns the best match.

    Args:
        chroma: Chroma feature matrix (12 x frames).

    Returns:
        (key_name, mode, confidence) e.g. ("C", "major", 0.82)
    """
    mean_chroma = chroma.mean(axis=1)

    best_score = -np.inf
    best_key = "C"
    best_mode = "major"

    for root in range(12):
        # Rotate profiles to match root
        maj_profile = np.roll(_MAJOR_PROFILE, root)
        min_profile = np.roll(_MINOR_PROFILE, root)

        maj_corr = float(np.corrcoef(mean_chroma, maj_profile)[0, 1])
        min_corr = float(np.corrcoef(mean_chroma, min_profile)[0, 1])

        if maj_corr > best_score:
            best_score = maj_corr
            best_key = NOTE_NAMES[root]
            best_mode = "major"

        if min_corr > best_score:
            best_score = min_corr
            best_key = NOTE_NAMES[root]
            best_mode = "minor"

    # Confidence: normalise the best correlation score to 0–1
    # Correlation ranges from -1 to 1; scale to 0–1
    confidence = float(np.clip((best_score + 1) / 2, 0.0, 1.0))

    return best_key, best_mode, confidence
