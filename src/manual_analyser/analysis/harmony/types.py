"""
analysis/harmony/types.py — Shared dataclasses for the harmony subpackage.

Kept in a separate module so that both __init__.py and sections.py can import
from here without creating a circular dependency.
"""

from dataclasses import dataclass, field


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
