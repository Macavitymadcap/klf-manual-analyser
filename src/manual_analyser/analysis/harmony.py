"""
analysis/harmony.py — Stage 3a: harmony analysis.

Responsibilities:
  - Detect musical key and mode (major/minor) from chroma features
  - Estimate chord progressions per section using template matching
  - Write initial section skeleton rows (label="unknown", boundaries from
    structural segmentation — see structure.py for boundary detection)
  - Write chord progressions per section
  - Update tracks table with key/mode/key_confidence

Note: chord detection accuracy is approximately 70-75% on modern recordings
and significantly lower on 1920s material. All downstream consumers of chord
data should treat it as approximate. See compatibility.md.

Writes to SQLite:
  UPDATE tracks SET key, mode, key_confidence WHERE track_id = ?

  INSERT INTO sections (track_id, position, start, end, duration,
    label, label_confidence, label_source) — skeleton rows only

  INSERT INTO chord_progressions (section_id, progression, chords_json)

The section skeleton is written here because chord estimation requires
section boundaries, and both are derived from the same chroma analysis.
Section boundaries come from structure.py (pass 1) if already written,
or fall back to equal-length segments.

Error handling (per docs/ERROR_HANDLING.md):
  - Numerical errors → write null for affected fields, log warning
  - Unhandled exception → write null for all fields, log error
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from manual_analyser.db import get_connection

logger = logging.getLogger(__name__)

# Chromatic scale note names
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

# Simple chord templates (major, minor, dominant 7th) in chroma space
# Each template is a 12-element binary vector
_CHORD_TEMPLATES = {}
for _root in range(12):
    _name = NOTE_NAMES[_root]
    # Major triad: root, major third, perfect fifth
    _maj = np.zeros(12)
    _maj[_root % 12] = 1
    _maj[(_root + 4) % 12] = 1
    _maj[(_root + 7) % 12] = 1
    _CHORD_TEMPLATES[_name] = _maj

    # Minor triad: root, minor third, perfect fifth
    _min = np.zeros(12)
    _min[_root % 12] = 1
    _min[(_root + 3) % 12] = 1
    _min[(_root + 7) % 12] = 1
    _CHORD_TEMPLATES[f"{_name}m"] = _min

    # Dominant 7th
    _dom7 = np.zeros(12)
    _dom7[_root % 12] = 1
    _dom7[(_root + 4) % 12] = 1
    _dom7[(_root + 7) % 12] = 1
    _dom7[(_root + 10) % 12] = 1
    _CHORD_TEMPLATES[f"{_name}7"] = _dom7


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChordEvent:
    """A single chord detection event."""

    start: float  # seconds
    end: float  # seconds
    chord: str  # e.g. "Am", "G7"


@dataclass
class SectionHarmony:
    """Harmony data for a single section."""

    section_id: int  # SQLite row id (set after INSERT)
    position: int
    start: float
    end: float
    progression: str  # compact string e.g. "Am - G - F - C"
    chords: list[ChordEvent]


@dataclass
class HarmonyResult:
    """Harmony analysis results for a single track."""

    key: str  # e.g. "C", "F#"
    mode: str  # "major" | "minor"
    key_confidence: float  # 0.0–1.0
    sections: list[SectionHarmony] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse_harmony(
    track_id: str,
    full_wav: Path,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> HarmonyResult | None:
    """
    Analyse harmony from the full mix WAV and write to SQLite.

    Reads existing section boundaries from the DB if structure.py has
    already run. Otherwise falls back to equal-length segments.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        full_wav: Path to the decoded mono WAV.
        data_dir: Root data directory (default: "data/").
        db_path: Path to SQLite database. Defaults to data/manual_analyser.db.

    Returns:
        HarmonyResult on success, or None if analysis failed.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        result = _compute_harmony(full_wav, short_id, resolved_db, track_id)
    except Exception as e:
        logger.exception("[%s] [harmony] Analysis failed: %s", short_id, e, exc_info=True)
        _write_nulls(resolved_db, track_id, short_id)
        return None

    _write_result(resolved_db, track_id, result, short_id)
    logger.info(
        "[%s] [harmony] key=%s %s confidence=%.2f sections=%d",
        short_id,
        result.key,
        result.mode,
        result.key_confidence,
        len(result.sections),
    )
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _compute_harmony(
    full_wav: Path,
    short_id: str,
    db_path: Path,
    track_id: str,
) -> HarmonyResult:
    """
    Load audio and compute harmony features.

    Args:
        full_wav: Path to the full mix WAV.
        short_id: First 8 chars for logging.
        db_path: SQLite path (to read existing section boundaries).
        track_id: Full track ID.

    Returns:
        HarmonyResult with all fields populated.
    """
    y, sr = librosa.load(str(full_wav), sr=None, mono=True)
    logger.debug("[%s] [harmony] Loaded %.1fs", short_id, len(y) / sr)

    # Compute chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Key detection
    key, mode, confidence = _detect_key(chroma)

    # Get section boundaries — from DB if available, else fallback
    boundaries = _get_section_boundaries(db_path, track_id, len(y) / sr)
    logger.debug("[%s] [harmony] Using %d sections", short_id, len(boundaries))

    # Chord estimation per section
    hop_length = 512  # default librosa hop
    sections = []
    for i, (start, end) in enumerate(boundaries):
        start_frame = librosa.time_to_frames(start, sr=sr, hop_length=hop_length)
        end_frame = librosa.time_to_frames(end, sr=sr, hop_length=hop_length)
        start_frame = max(0, min(start_frame, chroma.shape[1] - 1))
        end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))

        section_chroma = chroma[:, start_frame:end_frame]
        chords = _estimate_chords(section_chroma, start, sr, hop_length)
        progression = _chords_to_progression(chords)

        sections.append(
            SectionHarmony(
                section_id=-1,  # assigned after DB INSERT
                position=i,
                start=start,
                end=end,
                progression=progression,
                chords=chords,
            )
        )

    return HarmonyResult(
        key=key,
        mode=mode,
        key_confidence=confidence,
        sections=sections,
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


def _get_section_boundaries(
    db_path: Path,
    track_id: str,
    duration: float,
    n_fallback: int = 8,
) -> list[tuple[float, float]]:
    """
    Get section boundaries from the database.

    If structure.py has already written sections, use those boundaries.
    Otherwise, divide the track into n_fallback equal-length segments.

    Args:
        db_path: SQLite path.
        track_id: Full track ID.
        duration: Track duration in seconds.
        n_fallback: Number of equal segments to use if no sections exist.

    Returns:
        List of (start, end) tuples in seconds.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT start, end FROM sections WHERE track_id = ? ORDER BY position",
            (track_id,),
        ).fetchall()
    finally:
        conn.close()

    if rows:
        return [(row["start"], row["end"]) for row in rows]

    # Fallback: equal segments
    segment_len = duration / n_fallback
    return [(i * segment_len, min((i + 1) * segment_len, duration)) for i in range(n_fallback)]


def _estimate_chords(
    section_chroma: np.ndarray,
    section_start: float,
    sr: int,
    hop_length: int,
    min_chord_duration: float = 0.5,
) -> list[ChordEvent]:
    """
    Estimate chord sequence in a section using template matching.

    Divides the section chroma into short analysis windows and matches
    each window against the chord template library.

    Args:
        section_chroma: Chroma matrix for this section (12 x frames).
        section_start: Start time of section in seconds.
        sr: Sample rate.
        hop_length: Hop length used in chroma computation.
        min_chord_duration: Minimum chord duration to avoid noise.

    Returns:
        List of ChordEvent objects. May be empty if section is too short.
    """
    if section_chroma.shape[1] < 2:
        return []

    seconds_per_frame = hop_length / sr
    chords: list[ChordEvent] = []
    current_chord = None
    current_start = section_start

    for frame_idx in range(section_chroma.shape[1]):
        frame_chroma = section_chroma[:, frame_idx]
        chord_name = _match_chord(frame_chroma)
        frame_time = section_start + frame_idx * seconds_per_frame

        if chord_name != current_chord:
            if current_chord is not None:
                duration = frame_time - current_start
                if duration >= min_chord_duration:
                    chords.append(
                        ChordEvent(
                            start=round(current_start, 3),
                            end=round(frame_time, 3),
                            chord=current_chord,
                        )
                    )
            current_chord = chord_name
            current_start = frame_time

    # Close the last chord
    if current_chord is not None:
        end_time = section_start + section_chroma.shape[1] * seconds_per_frame
        duration = end_time - current_start
        if duration >= min_chord_duration:
            chords.append(
                ChordEvent(
                    start=round(current_start, 3),
                    end=round(end_time, 3),
                    chord=current_chord,
                )
            )

    return chords


def _match_chord(chroma_frame: np.ndarray) -> str:
    """
    Match a single chroma frame to the best chord template.

    Uses dot product similarity (equivalent to cosine similarity for
    binary templates against normalised chroma).

    Args:
        chroma_frame: 12-element chroma vector.

    Returns:
        Chord name string e.g. "Am", "G", "C7".
    """
    norm = np.linalg.norm(chroma_frame)
    if norm == 0:
        return "C"  # silence defaults to C

    normalised = chroma_frame / norm
    best_chord = "C"
    best_score = -np.inf

    for chord_name, template in _CHORD_TEMPLATES.items():
        score = float(np.dot(normalised, template))
        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord


def _chords_to_progression(chords: list[ChordEvent]) -> str:
    """
    Summarise a chord sequence as a compact progression string.

    Deduplicates consecutive identical chords and returns the unique
    sequence joined with " - ".

    Args:
        chords: List of ChordEvent objects.

    Returns:
        Progression string e.g. "Am - G - F - C", or "unknown" if empty.
    """
    if not chords:
        return "unknown"

    unique = []
    prev = None
    for event in chords:
        if event.chord != prev:
            unique.append(event.chord)
            prev = event.chord

    # Limit to 8 unique chords for readability
    return " - ".join(unique[:8])


# ---------------------------------------------------------------------------
# SQLite writes
# ---------------------------------------------------------------------------


def _write_result(
    db_path: Path,
    track_id: str,
    result: HarmonyResult,
    short_id: str,
) -> None:
    """Write HarmonyResult to tracks, sections, and chord_progressions tables."""
    conn = get_connection(db_path)
    try:
        with conn:
            # Update key/mode on tracks row
            conn.execute(
                """
                UPDATE tracks SET key = ?, mode = ?, key_confidence = ?
                WHERE track_id = ?
                """,
                (result.key, result.mode, round(result.key_confidence, 4), track_id),
            )

            # Check if sections already exist (written by structure.py)
            existing = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE track_id = ?",
                (track_id,),
            ).fetchone()[0]

            for section in result.sections:
                if existing == 0:
                    # Write skeleton section row
                    cursor = conn.execute(
                        """
                        INSERT INTO sections
                            (track_id, position, start, end, duration,
                             label, label_confidence, label_source)
                        VALUES (?, ?, ?, ?, ?, 'unknown', 0.0, 'acoustic')
                        """,
                        (
                            track_id,
                            section.position,
                            round(section.start, 3),
                            round(section.end, 3),
                            round(section.end - section.start, 3),
                        ),
                    )
                    section_id = cursor.lastrowid
                else:
                    # Sections exist — find the matching row by position
                    row = conn.execute(
                        "SELECT id FROM sections WHERE track_id = ? AND position = ?",
                        (track_id, section.position),
                    ).fetchone()
                    section_id = row["id"] if row else None

                if section_id is None:
                    continue

                # Write chord progression
                chords_json = json.dumps([{"start": c.start, "end": c.end, "chord": c.chord} for c in section.chords])
                conn.execute(
                    """
                    INSERT INTO chord_progressions (section_id, progression, chords_json)
                    VALUES (?, ?, ?)
                    """,
                    (section_id, section.progression, chords_json),
                )

    finally:
        conn.close()


def _write_nulls(db_path: Path, track_id: str, short_id: str) -> None:
    """Write null for harmony fields when analysis fails."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET key = NULL, mode = NULL, key_confidence = NULL
                WHERE track_id = ?
                """,
                (track_id,),
            )
        logger.warning("[%s] [harmony] Wrote null fields due to failure", short_id)
    finally:
        conn.close()
