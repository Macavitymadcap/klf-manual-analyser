"""
analysis/structure/labelling.py — Section label assignment.

Provides _assign_labels(), which converts raw section data (boundaries,
energy profile, lyric features) into a labelled list of SectionLabel objects.

The algorithm runs eight heuristic steps in priority order:

  1. Cross-section phrase detection  — identify hook phrases
  2. Intro detection                 — pre-vocal or quiet opening
  3. Outro detection                 — post-vocal or quiet close
  4. Chorus detection                — high repetition + above-average energy
  5. Breakdown detection             — lowest-energy section in second half
  6. Double chorus detection         — high-energy section after breakdown
  7. Pre-chorus detection            — rising energy immediately before chorus
  8. Verse labelling                 — remaining sections with lyric content

State is carried through steps via _LabelState, which holds the mutable
working arrays (labels, confidences, sources) alongside read-only inputs
that every step may need. Steps mutate _LabelState in place and return
nothing; _assign_labels assembles the final SectionLabel list at the end.

Only _assign_labels is part of the public API. The step functions and
_LabelState are internal; they are not re-exported from structure/__init__.py.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from manual_analyser.analysis.structure.types import (
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    SectionLabel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Working state container
# ---------------------------------------------------------------------------


@dataclass
class _LabelState:
    """
    Mutable working state shared across all labelling steps.

    Read-only inputs are stored alongside the mutable working arrays so
    each step function receives a single argument rather than six.
    """

    # Read-only inputs
    sections: list[dict]
    energies: list[float]
    lyric_data: list[dict]
    duration: float
    short_id: str
    n: int

    # Derived read-only (computed once in _assign_labels before steps run)
    mean_energy: float = 0.0
    repetition_scores: list[int] = field(default_factory=list)
    cross_section_phrases: set[str] = field(default_factory=set)

    # Mutable working arrays — one entry per section
    labels: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    # Indices written by earlier steps, consumed by later ones
    chorus_indices: list[int] = field(default_factory=list)
    breakdown_idx: int | None = None

    def __post_init__(self) -> None:
        if not self.labels:
            self.labels = ["unknown"] * self.n
        if not self.confidences:
            self.confidences = [LOW_CONFIDENCE] * self.n
        if not self.sources:
            self.sources = ["acoustic"] * self.n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _assign_labels(
    sections: list[dict],
    energies: list[float],
    lyric_data: list[dict],
    duration: float,
    short_id: str,
) -> list[SectionLabel]:
    """
    Apply heuristic labelling to produce a list of SectionLabels.

    Args:
        sections: Section dicts with 'id', 'pos', 'start', 'end'.
        energies: Mean normalised RMS energy per section (0.0–1.0).
        lyric_data: Per-section dicts with 'lyric_density', 'word_count',
            'phrases' (Counter), 'repeated_phrase'.
        duration: Total track duration in seconds.
        short_id: First 8 chars of track_id for log messages.

    Returns:
        List of SectionLabel objects, one per section, in position order.
    """
    n = len(sections)
    if n == 0:
        return []

    state = _LabelState(
        sections=sections,
        energies=energies,
        lyric_data=lyric_data,
        duration=duration,
        short_id=short_id,
        n=n,
        mean_energy=float(np.mean(energies)),
    )

    _step1_find_cross_section_phrases(state)
    _step2_identify_intro(state)
    _step3_identify_outro(state)
    _step4_identify_choruses(state)
    _step5_identify_breakdown(state)
    _step6_identify_double_chorus(state)
    _step7_identify_pre_chorus(state)
    _step8_label_verses(state)

    return _build_result(state)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def _step1_find_cross_section_phrases(state: _LabelState) -> None:
    """
    Step 1 — Identify phrases that recur across multiple sections.

    A phrase appearing in 2+ sections is a strong chorus signal. The
    per-section repetition score (count of cross-section phrase occurrences)
    is stored in state for use by step 4.
    """
    phrase_section_counts: Counter = Counter()
    for ld in state.lyric_data:
        for phrase, count in ld["phrases"].items():
            if count > 0:
                phrase_section_counts[phrase] += 1

    state.cross_section_phrases = {p for p, c in phrase_section_counts.items() if c >= 2}

    state.repetition_scores = [
        sum(ld["phrases"][p] for p in state.cross_section_phrases if p in ld["phrases"]) for ld in state.lyric_data
    ]


def _step2_identify_intro(state: _LabelState) -> None:
    """
    Step 2 — Label the first section as intro if it precedes vocal activity.

    Conditions (either is sufficient):
    - Very low lyric density (< 0.1) — pre-vocal
    - Early in the track (ends before 15% of duration) AND below-average energy
    """
    if state.n < 2:
        return

    first_lyric = state.lyric_data[0]["lyric_density"]
    is_early = state.sections[0]["end"] <= state.duration * 0.15
    is_quiet = state.energies[0] < state.mean_energy * 0.9

    if first_lyric < 0.1 or (is_early and is_quiet):
        state.labels[0] = "intro"
        # High confidence if near-silent, medium if just early and quiet
        state.confidences[0] = HIGH_CONFIDENCE if first_lyric < 0.05 else MEDIUM_CONFIDENCE
        state.sources[0] = "hybrid" if first_lyric < 0.05 else "acoustic"


def _step3_identify_outro(state: _LabelState) -> None:
    """
    Step 3 — Label the last section as outro if vocal activity has ended.

    Conditions (both required):
    - Starts after 85% of duration
    - Low lyric density OR below-average energy
    """
    if state.n < 2:
        return

    last_lyric = state.lyric_data[-1]["lyric_density"]
    is_late = state.sections[-1]["start"] >= state.duration * 0.85
    is_quiet = state.energies[-1] < state.mean_energy * 0.9

    if is_late and (last_lyric < 0.1 or is_quiet):
        state.labels[-1] = "outro"
        state.confidences[-1] = MEDIUM_CONFIDENCE
        state.sources[-1] = "acoustic"


def _step4_identify_choruses(state: _LabelState) -> None:
    """
    Step 4 — Label sections with high phrase repetition as choruses.

    A section is a chorus candidate if its repetition score is >= 70% of
    the maximum repetition score across all sections (strong signal), or
    >= 50% with above-average energy (weaker signal, lyric source only).

    Sets state.chorus_indices for use by steps 6 and 7.
    """
    max_rep = max(state.repetition_scores) if state.repetition_scores else 0
    if max_rep == 0:
        return

    for i in range(state.n):
        if state.labels[i] != "unknown":
            continue

        rep_norm = state.repetition_scores[i] / max_rep
        energy_above_avg = state.energies[i] >= state.mean_energy * 0.95

        if rep_norm >= 0.7 and energy_above_avg:
            state.labels[i] = "chorus"
            state.confidences[i] = HIGH_CONFIDENCE
            state.sources[i] = "hybrid"
            state.chorus_indices.append(i)
        elif rep_norm >= 0.5:
            state.labels[i] = "chorus"
            state.confidences[i] = MEDIUM_CONFIDENCE
            state.sources[i] = "lyric"
            state.chorus_indices.append(i)


def _step5_identify_breakdown(state: _LabelState) -> None:
    """
    Step 5 — Label the lowest-energy unlabelled section in the second half.

    The breakdown must be meaningfully below average energy (< 70% of mean)
    to qualify. Sets state.breakdown_idx for use by step 6.
    """
    second_half_start = state.n // 2
    candidates = [i for i in range(second_half_start, state.n) if state.labels[i] == "unknown"]

    if not candidates:
        return

    min_idx = min(candidates, key=lambda i: state.energies[i])

    if state.energies[min_idx] < state.mean_energy * 0.7:
        state.labels[min_idx] = "breakdown"
        state.confidences[min_idx] = MEDIUM_CONFIDENCE
        state.sources[min_idx] = "acoustic"
        state.breakdown_idx = min_idx


def _step6_identify_double_chorus(state: _LabelState) -> None:
    """
    Step 6 — Label the first high-energy unlabelled section after the breakdown.

    Only the section immediately following the breakdown is considered.
    Must be at or above average energy to qualify.
    """
    if state.breakdown_idx is None:
        return

    for j in range(state.breakdown_idx + 1, state.n):
        if state.labels[j] == "unknown":
            if state.energies[j] >= state.mean_energy * 1.05:
                state.labels[j] = "double_chorus"
                state.confidences[j] = MEDIUM_CONFIDENCE
                state.sources[j] = "acoustic"
            break  # only check the immediate next unlabelled section


def _step7_identify_pre_chorus(state: _LabelState) -> None:
    """
    Step 7 — Label rising-energy unlabelled sections immediately before a chorus.

    A section qualifies as pre-chorus if:
    - The next section is labelled chorus
    - Its energy is rising relative to the previous section (> 5% uplift)
    """
    for i in range(1, state.n - 1):
        if state.labels[i] != "unknown":
            continue

        next_is_chorus = i + 1 < state.n and state.labels[i + 1] == "chorus"
        energy_rising = state.energies[i] > state.energies[max(0, i - 1)] * 1.05

        if next_is_chorus and energy_rising:
            state.labels[i] = "pre_chorus"
            state.confidences[i] = LOW_CONFIDENCE
            state.sources[i] = "acoustic"


def _step8_label_verses(state: _LabelState) -> None:
    """
    Step 8 — Label remaining sections with lyric content as verses.

    Any still-unknown section with lyric density > 0.05 is a verse.
    Source is 'lyric' if density is strong (> 0.2), else 'acoustic'.
    Sections below the lyric threshold remain 'unknown'.
    """
    for i in range(state.n):
        if state.labels[i] != "unknown":
            continue

        if state.lyric_data[i]["lyric_density"] > 0.05:
            state.labels[i] = "verse"
            state.confidences[i] = LOW_CONFIDENCE
            state.sources[i] = "lyric" if state.lyric_data[i]["lyric_density"] > 0.2 else "acoustic"


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _build_result(state: _LabelState) -> list[SectionLabel]:
    """Assemble SectionLabel objects from the completed _LabelState."""
    return [
        SectionLabel(
            position=i,
            start=state.sections[i]["start"],
            end=state.sections[i]["end"],
            label=state.labels[i],
            label_confidence=state.confidences[i],
            label_source=state.sources[i],
            mean_energy=state.energies[i],
            lyric_density=state.lyric_data[i]["lyric_density"],
            repeated_phrase=state.lyric_data[i]["repeated_phrase"],
        )
        for i, _ in enumerate(state.sections)
    ]
