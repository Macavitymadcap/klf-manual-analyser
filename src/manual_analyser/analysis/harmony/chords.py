# Simple chord templates (major, minor, dominant 7th) in chroma space
# Each template is a 12-element binary vector
import numpy as np

from manual_analyser.analysis.harmony.keys import NOTE_NAMES
from manual_analyser.analysis.harmony.types import ChordEvent

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
