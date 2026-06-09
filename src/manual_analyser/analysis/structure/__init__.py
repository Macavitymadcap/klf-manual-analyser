"""
analysis/structure — Section boundary detection and label alignment.

Public API:
    segment_track(track_id, full_wav, data_dir, db_path) -> list[tuple]
    align_sections(track_id, data_dir, db_path) -> list[SectionLabel] | None

Submodules:
    types.py      — SectionLabel dataclass and confidence constants
    boundaries.py — pass 1: librosa segmentation, skeleton writes
    alignment.py  — pass 2: DB reads, feature computation, label writes
    labelling.py  — _assign_labels and one function per labelling step

All names that tests import from manual_analyser.analysis.structure are
re-exported here, so no test files need updating.
"""

# Types — import from types.py to avoid circular imports
# Pass 2 — public API and helpers the tests import directly
from manual_analyser.analysis.structure.alignment import (
    _compute_lyric_features,
    _compute_section_energies,
    _find_repeated_phrase,
    _run_alignment,
    _write_labels,
    align_sections,
)

# Pass 1
from manual_analyser.analysis.structure.boundaries import segment_track

# Labelling — _assign_labels is imported by tests via this package
from manual_analyser.analysis.structure.labelling import _assign_labels
from manual_analyser.analysis.structure.types import SectionLabel

__all__ = [
    # Types
    "SectionLabel",
    # Pass 1
    "segment_track",
    # Pass 2
    "align_sections",
    "_run_alignment",
    "_compute_section_energies",
    "_compute_lyric_features",
    "_find_repeated_phrase",
    "_write_labels",
    # Labelling
    "_assign_labels",
]
