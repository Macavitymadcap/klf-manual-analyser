# Number of segments to detect (default; adjusted for track length)
import logging
from pathlib import Path

import librosa
import numpy as np

from manual_analyser.db import get_connection

logger = logging.Logger(__name__)

DEFAULT_N_SEGMENTS = 8
MIN_SEGMENT_DURATION = 5.0  # seconds — segments shorter than this are merged
MIN_SEGMENTS = 4
MAX_SEGMENTS = 12


def segment_track(
    track_id: str,
    full_wav: Path,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> list[tuple[float, float]]:
    """
    Detect section boundaries using librosa agglomerative segmentation.

    Writes skeleton section rows (label='unknown') to SQLite only if no
    sections already exist for this track. This allows harmony.py to write
    sections first if it runs before structure.py.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        full_wav: Path to the decoded mono WAV.
        data_dir: Root data directory.
        db_path: Path to SQLite database.

    Returns:
        List of (start, end) boundary tuples in seconds.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        boundaries = _detect_boundaries(full_wav, short_id)
    except Exception as e:
        logger.exception("[%s] [structure/pass1] Boundary detection failed: %s", short_id, e, exc_info=True)
        return []

    _write_section_skeletons(resolved_db, track_id, boundaries, short_id, logger)
    logger.info("[%s] [structure/pass1] Detected %d sections", short_id, len(boundaries))
    return boundaries


def _detect_boundaries(full_wav: Path, short_id: str) -> list[tuple[float, float]]:
    """
    Use librosa agglomerative segmentation to find section boundaries.

    Combines chroma and MFCC features for segmentation, which captures
    both harmonic and timbral changes.

    Args:
        full_wav: Path to full mix WAV.
        short_id: For log messages.

    Returns:
        List of (start, end) tuples in seconds.
    """
    y, sr = librosa.load(str(full_wav), sr=None, mono=True)
    duration = len(y) / sr
    logger.debug("[%s] [structure/pass1] Loaded %.1fs", short_id, duration)

    # Scale number of segments to track length
    n_segments = max(MIN_SEGMENTS, min(MAX_SEGMENTS, int(duration / 20)))

    # Combine chroma and MFCC features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.vstack([chroma, mfcc])

    # Agglomerative segmentation
    bounds_frames = librosa.segment.agglomerative(features, k=n_segments)
    bound_times = librosa.frames_to_time(bounds_frames, sr=sr)

    # Include track endpoints
    all_times = np.concatenate([[0.0], bound_times, [duration]])
    all_times = np.unique(np.clip(all_times, 0.0, duration))

    # Build boundary pairs
    boundaries = []
    for i in range(len(all_times) - 1):
        start = float(all_times[i])
        end = float(all_times[i + 1])
        if end - start >= MIN_SEGMENT_DURATION:
            boundaries.append((start, end))

    # If too few segments after filtering, fall back to equal division
    if len(boundaries) < MIN_SEGMENTS:
        logger.warning("[%s] [structure/pass1] Too few segments after filtering, using equal split", short_id)
        seg_len = duration / DEFAULT_N_SEGMENTS
        boundaries = [(i * seg_len, min((i + 1) * seg_len, duration)) for i in range(DEFAULT_N_SEGMENTS)]

    return boundaries


def _write_section_skeletons(
    db_path: Path,
    track_id: str,
    boundaries: list[tuple[float, float]],
    short_id: str,
    logger: logging.Logger,
) -> None:
    """
    Write skeleton section rows if none exist for this track.

    Skips writing if sections already exist (written by harmony.py).
    """
    conn = get_connection(db_path)
    try:
        existing = conn.execute("SELECT COUNT(*) FROM sections WHERE track_id = ?", (track_id,)).fetchone()[0]

        if existing > 0:
            logger.debug(
                "[%s] [structure/pass1] Sections already exist (%d), skipping skeleton write",
                short_id,
                existing,
            )
            return

        with conn:
            for i, (start, end) in enumerate(boundaries):
                conn.execute(
                    """
                    INSERT INTO sections
                        (track_id, position, start, end, duration,
                         label, label_confidence, label_source)
                    VALUES (?, ?, ?, ?, ?, 'unknown', 0.0, 'acoustic')
                    """,
                    (track_id, i, round(start, 3), round(end, 3), round(end - start, 3)),
                )
    finally:
        conn.close()
