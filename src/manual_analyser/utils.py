"""
utils.py — shared utilities used across the manual_analyser package.

Modules that need these functions should import directly:
    from manual_analyser.utils import get_torch_device, make_track_id

Do not import this module as a side-effect-only module — all functions
are pure and stateless.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_torch_device() -> str:
    """
    Return the best available torch device: cuda > mps > cpu.

    Checks in priority order:
    - CUDA (Fedora / NVIDIA RTX — primary production environment)
    - MPS (macOS Apple Silicon — development environment)
    - CPU (fallback)

    Import is deferred to avoid requiring torch at module load time
    in contexts where it isn't needed (e.g. criteria loading, reporting).
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Track identity
# ---------------------------------------------------------------------------


def make_track_id(path: Path | str) -> str:
    """
    Return a stable 32-character MD5 hex digest identifying a track.

    The digest is computed from the absolute path string, so the same file
    at the same location always produces the same ID. Moving or renaming
    the file produces a different ID — this is intentional, as the pipeline
    caches stems at data/stems/{track_id}/ and a renamed file should be
    treated as a new track.

    Args:
        path: Path to the MP3 file.

    Returns:
        32-character lowercase hex string.
    """
    abs_path = str(Path(path).resolve())
    return hashlib.md5(abs_path.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Expected format: Artist_Name-Song_Title.mp3
# Underscores separate words within artist or title.
# The hyphen separates artist from title (split on first hyphen only).
_FILENAME_PATTERN = re.compile(r"^(?P<artist>[^-]+)-(?P<title>.+)$")


def parse_filename(path: Path | str) -> tuple[str | None, str | None]:
    """
    Parse artist and song title from a filename following the convention:
        Artist_Name-Song_Title.mp3

    Returns:
        (artist, song_name) — both title-cased with underscores replaced
        by spaces. Returns (None, None) if the filename does not match
        the expected pattern; the caller should log a warning.

    Examples:
        >>> parse_filename("The_KLF-Doctorin_The_Tardis.mp3")
        ('The Klf', 'Doctorin The Tardis')
        >>> parse_filename("Louis_Armstrong-Heebie_Jeebies.mp3")
        ('Louis Armstrong', 'Heebie Jeebies')
        >>> parse_filename("unknown_file.mp3")
        (None, None)
    """
    stem = Path(path).stem  # remove .mp3
    match = _FILENAME_PATTERN.match(stem)
    if not match:
        return None, None

    artist = match.group("artist").replace("_", " ").title()
    title = match.group("title").replace("_", " ").title()
    return artist, title


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Beat pattern encoding
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

    # Compute median inter-beat interval in frames
    ibi = float(np.median(np.diff(beat_frames)))
    step_size = ibi / (steps / 4)  # ibi covers 4 steps (one beat = 4 16th notes)

    # Collect patterns per bar
    bar_patterns: list[list[int]] = []
    bar_length = ibi * 4  # 4 beats per bar

    # Align to first beat
    start = beat_frames[0]

    # Find approximate bar boundaries
    n_bars = max(1, int((beat_frames[-1] - start) / bar_length))

    for bar_idx in range(n_bars):
        bar_start = start + bar_idx * bar_length
        bar_end = bar_start + bar_length
        pattern = [0] * steps

        # Find onsets within this bar
        bar_onsets = onset_frames[(onset_frames >= bar_start) & (onset_frames < bar_end)]

        for onset in bar_onsets:
            step = int((onset - bar_start) / step_size)
            if 0 <= step < steps:
                pattern[step] = 1

        bar_patterns.append(pattern)

    if not bar_patterns:
        return "0" * steps

    # Return modal pattern (most common value at each step position)
    patterns_array = np.array(bar_patterns)
    modal = (patterns_array.mean(axis=0) >= 0.5).astype(int)
    return "".join(str(v) for v in modal)


# ---------------------------------------------------------------------------
# Groove feel classification
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
        sr: Sample rate (used for context, not direct calculation here).
        hop_length: Hop length used during beat tracking.

    Returns:
        'straight', 'swung', or 'unclear'.
    """
    if len(beat_times) < 4:
        return "unclear"

    # Estimate 8th-note positions as midpoints between beats
    ibis = np.diff(beat_times)
    if len(ibis) == 0:
        return "unclear"

    # For each pair of consecutive beats, find the midpoint (8th note)
    # and compute the ratio: first half / total beat duration
    # In a straight feel, midpoints fall exactly at 0.5 of the beat.
    # In a swung feel, the first subdivision is longer (ratio > 0.5).
    #
    # We approximate swing ratio from the inter-beat interval stability
    # and any detectable subdivision unevenness via beat tracking confidence.
    #
    # Simple heuristic: compute coefficient of variation of IBIs.
    # Highly regular IBIs → straight. Systematic long-short alternation → swung.
    # This is a coarse approximation; refine during implementation if needed.

    median_ibi = float(np.median(ibis))
    if median_ibi == 0:
        return "unclear"

    # Detect long-short alternation pattern (characteristic of swing)
    normalised = ibis / median_ibi
    even = normalised[0::2]  # even-indexed IBIs
    odd = normalised[1::2]  # odd-indexed IBIs

    if len(even) == 0 or len(odd) == 0:
        return "unclear"

    ratio = float(np.mean(even) / (np.mean(even) + np.mean(odd)))

    if ratio > 0.55:
        return "swung"
    if ratio < 0.52:
        return "straight"
    return "unclear"
