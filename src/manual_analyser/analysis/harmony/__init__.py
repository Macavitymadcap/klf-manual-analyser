"""
analysis/harmony — Harmony feature analysis for the KLF Manual Analyser.

Public API:
    analyse_harmony(track_id, full_wav, data_dir, db_path) -> HarmonyResult | None

Submodules:
    types.py    — shared dataclasses (ChordEvent, SectionHarmony, HarmonyResult)
    keys.py     — key/mode detection (Krumhansl-Schmuckler profiles)
    chords.py   — chord template matching and progression formatting
    sections.py — section boundary helpers and SQLite writes

All names that tests import from manual_analyser.analysis.harmony are
re-exported here, so no test files need updating.
"""

import logging
from pathlib import Path

import librosa

from manual_analyser.analysis.harmony.chords import (
    _chords_to_progression,
    _estimate_chords,
    _match_chord,
)
from manual_analyser.analysis.harmony.keys import _detect_key
from manual_analyser.analysis.harmony.sections import (
    _get_section_boundaries,
    _write_nulls,
    _write_result,
)

# Import types from types.py — NOT from this __init__ — to avoid circular imports.
# sections.py also imports from types.py for the same reason.
from manual_analyser.analysis.harmony.types import (
    ChordEvent,
    HarmonyResult,
    SectionHarmony,
)

logger = logging.getLogger(__name__)


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
        logger.error("[%s] [harmony] Analysis failed: %s", short_id, e, exc_info=True)
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
# Analysis orchestration
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

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    key, mode, confidence = _detect_key(chroma)

    boundaries = _get_section_boundaries(db_path, track_id, len(y) / sr)
    logger.debug("[%s] [harmony] Using %d sections", short_id, len(boundaries))

    hop_length = 512
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
                section_id=-1,
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


# ---------------------------------------------------------------------------
# Re-exports for test compatibility
# Tests import all of these from manual_analyser.analysis.harmony directly.
# ---------------------------------------------------------------------------

__all__ = [
    # Types
    "ChordEvent",
    "SectionHarmony",
    "HarmonyResult",
    # Public API
    "analyse_harmony",
    # Private functions imported by tests
    "_compute_harmony",
    "_detect_key",
    "_estimate_chords",
    "_chords_to_progression",
    "_match_chord",
    "_get_section_boundaries",
    "_write_result",
    "_write_nulls",
]
