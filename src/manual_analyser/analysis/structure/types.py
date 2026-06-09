"""
analysis/structure/types.py — Shared types for the structure subpackage.

Kept separate so that labelling.py, alignment.py, and __init__.py can all
import from here without circular dependencies.
"""

from dataclasses import dataclass

# Confidence thresholds used by the labelling heuristics
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.3


@dataclass
class SectionLabel:
    """A labelled section produced by the alignment pass."""

    position: int
    start: float
    end: float
    label: str  # "intro" | "verse" | "pre_chorus" | "chorus" |
    # "breakdown" | "double_chorus" | "bridge" |
    # "outro" | "unknown"
    label_confidence: float  # 0.0–1.0
    label_source: str  # "acoustic" | "lyric" | "hybrid"
    mean_energy: float  # normalised 0.0–1.0
    lyric_density: float  # normalised 0.0–1.0
    repeated_phrase: str | None
