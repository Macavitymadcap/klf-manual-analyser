"""
analysis/structure.py — Stages 3a (pass 1) and 4 (pass 2) of the pipeline.

Pass 1 (segment_track): Detect section boundaries using librosa agglomerative
  segmentation. Writes boundary-only section rows to SQLite with label="unknown".
  Runs as part of Stage 3a alongside other analysis modules.

Pass 2 (align_sections): Hybrid alignment — reads section boundaries, RMS energy
  profile, and transcript timestamps from SQLite. Cross-references all three
  signals to assign section labels with confidence scores. Updates the sections
  table with labels, confidence, label_source, mean_energy, lyric_density,
  and repeated_phrase.

Note: msaf is confirmed broken on Python 3.11+ (scipy.inf removed in scipy 1.11).
This module uses librosa.segment.agglomerative exclusively.
See klf-mir-dev/references/compatibility.md.

Writes to SQLite (pass 1):
  INSERT INTO sections (track_id, position, start, end, duration,
    label, label_confidence, label_source)
  — only if no sections exist yet for this track

Writes to SQLite (pass 2):
  UPDATE sections SET label, label_confidence, label_source,
    mean_energy, lyric_density, repeated_phrase
  WHERE track_id = ? AND position = ?

Error handling (per docs/ERROR_HANDLING.md):
  - Pass 1 failure → leave sections table empty; pass 2 will handle gracefully
  - Pass 2 failure → leave labels as "unknown"; scoring will handle gracefully
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from manual_analyser.db import get_connection
from manual_analyser.utils import normalise_lyric_density

logger = logging.getLogger(__name__)

# Number of segments to detect (default; adjusted for track length)
DEFAULT_N_SEGMENTS = 8
MIN_SEGMENT_DURATION = 5.0  # seconds — segments shorter than this are merged
MIN_SEGMENTS = 4
MAX_SEGMENTS = 12

# Confidence thresholds
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SectionLabel:
    """A labelled section after the alignment pass."""

    position: int
    start: float
    end: float
    label: str
    label_confidence: float
    label_source: str  # "acoustic" | "lyric" | "hybrid"
    mean_energy: float
    lyric_density: float
    repeated_phrase: str | None


# ---------------------------------------------------------------------------
# Pass 1: Boundary detection
# ---------------------------------------------------------------------------


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

    _write_section_skeletons(resolved_db, track_id, boundaries, short_id)
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


# ---------------------------------------------------------------------------
# Pass 2: Hybrid section alignment
# ---------------------------------------------------------------------------


def align_sections(
    track_id: str,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> list[SectionLabel] | None:
    """
    Assign section labels by cross-referencing acoustic and lyric signals.

    Reads from SQLite:
    - Section boundaries (from pass 1 or harmony.py)
    - RMS energy profile (from energy.py)
    - Transcript segments (from whisper.py)

    Labelling heuristics (applied in priority order):
    1. Intro: first section(s) before vocal activity begins, or very short opening
    2. Outro: last section(s) after main vocal activity ends
    3. Chorus: section(s) with highest lyric repetition (most-repeated phrase recurs)
    4. Breakdown: lowest-energy section in the second half of the track
    5. Double chorus: high-energy section immediately following the breakdown
    6. Pre-chorus: section before a chorus with rising energy
    7. Verse: remaining labelled sections
    8. Unknown: anything with insufficient confidence

    Confidence scores reflect signal agreement:
    - "hybrid": both acoustic and lyric signals agree → HIGH_CONFIDENCE
    - "lyric": lyric signal only → MEDIUM_CONFIDENCE
    - "acoustic": acoustic signal only → LOW_CONFIDENCE

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        data_dir: Root data directory.
        db_path: Path to SQLite database.

    Returns:
        List of SectionLabel objects, or None if alignment failed.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        labels = _run_alignment(resolved_db, track_id, short_id)
    except Exception as e:
        logger.exception("[%s] [structure/pass2] Alignment failed: %s", short_id, e, exc_info=True)
        return None

    if labels:
        _write_labels(resolved_db, track_id, labels, short_id)
        logger.info(
            "[%s] [structure/pass2] Labelled %d sections: %s",
            short_id,
            len(labels),
            [label.label for label in labels],
        )

    return labels


def _run_alignment(
    db_path: Path,
    track_id: str,
    short_id: str,
) -> list[SectionLabel]:
    """
    Core alignment logic — reads all data from DB and produces labels.

    Args:
        db_path: SQLite path.
        track_id: Full track ID.
        short_id: For log messages.

    Returns:
        List of SectionLabel objects.
    """
    conn = get_connection(db_path)
    try:
        # Load sections
        section_rows = conn.execute(
            "SELECT id, position, start, end FROM sections WHERE track_id = ? ORDER BY position",
            (track_id,),
        ).fetchall()

        if not section_rows:
            logger.warning("[%s] [structure/pass2] No sections found", short_id)
            return []

        # Load RMS profile
        ts_row = conn.execute(
            "SELECT rms_profile_json FROM tracks_timeseries WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        rms_profile = json.loads(ts_row["rms_profile_json"]) if ts_row else []

        # Load transcript segments
        transcript_rows = conn.execute(
            "SELECT start, end, text FROM transcript_segments WHERE track_id = ? ORDER BY start",
            (track_id,),
        ).fetchall()

        track_duration = conn.execute("SELECT duration FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
        duration = float(track_duration["duration"]) if track_duration else 0.0

    finally:
        conn.close()

    # Convert to working structures
    sections = [{"id": r["id"], "pos": r["position"], "start": r["start"], "end": r["end"]} for r in section_rows]

    # Compute per-section features
    rms_array = np.array(rms_profile) if rms_profile else np.array([])
    rms_per_section = _compute_section_energies(sections, rms_array, duration)
    lyric_data = _compute_lyric_features(sections, transcript_rows)

    # Run labelling heuristics
    labels = _assign_labels(sections, rms_per_section, lyric_data, duration, short_id)

    return labels


def _compute_section_energies(
    sections: list[dict],
    rms_array: np.ndarray,
    duration: float,
    rms_interval: float = 0.5,
) -> list[float]:
    """
    Compute mean normalised RMS energy for each section.

    Args:
        sections: List of section dicts with start/end.
        rms_array: Normalised RMS profile (one value per rms_interval seconds).
        duration: Total track duration in seconds.
        rms_interval: Sampling interval of RMS profile.

    Returns:
        List of mean energy values (0.0–1.0) per section.
    """
    if len(rms_array) == 0:
        return [0.5] * len(sections)

    energies = []
    for sec in sections:
        start_idx = int(sec["start"] / rms_interval)
        end_idx = int(sec["end"] / rms_interval) + 1
        start_idx = max(0, min(start_idx, len(rms_array) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(rms_array)))
        section_rms = rms_array[start_idx:end_idx]
        energies.append(float(np.mean(section_rms)) if len(section_rms) > 0 else 0.5)

    return energies


def _compute_lyric_features(
    sections: list[dict],
    transcript_rows: list,
) -> list[dict]:
    """
    Compute lyric density and repeated phrase for each section.

    Args:
        sections: List of section dicts.
        transcript_rows: Rows from transcript_segments table.

    Returns:
        List of dicts with 'lyric_density', 'word_count', 'phrases' per section.
    """
    results = []

    for sec in sections:
        duration = sec["end"] - sec["start"]
        # Find transcript segments overlapping this section
        words_in_section = []
        for row in transcript_rows:
            # Include segment if it overlaps the section
            if row["end"] > sec["start"] and row["start"] < sec["end"]:
                words = row["text"].strip().split()
                words_in_section.extend(words)

        word_count = len(words_in_section)
        raw_density = word_count / max(duration, 1.0)
        lyric_density = normalise_lyric_density(raw_density)

        # Find most repeated phrase (2-3 word n-grams)
        repeated_phrase = _find_repeated_phrase(words_in_section)

        results.append(
            {
                "lyric_density": lyric_density,
                "word_count": word_count,
                "phrases": _extract_phrases(words_in_section),
                "repeated_phrase": repeated_phrase,
            }
        )

    return results


def _extract_phrases(words: list[str], n: int = 3) -> Counter:
    """Extract n-gram phrase counts from a word list."""
    if len(words) < n:
        return Counter()
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return Counter(ngrams)


def _find_repeated_phrase(words: list[str]) -> str | None:
    """Find the most repeated 3-word phrase, or None if no repetition."""
    phrases = _extract_phrases(words, n=3)
    if not phrases:
        return None
    most_common = phrases.most_common(1)[0]
    # Only return if the phrase appears at least twice
    return most_common[0] if most_common[1] >= 2 else None


def _assign_labels(
    sections: list[dict],
    energies: list[float],
    lyric_data: list[dict],
    duration: float,
    short_id: str,
) -> list[SectionLabel]:
    """
    Apply labelling heuristics to produce a list of SectionLabels.

    Heuristics applied in priority order:
    1. Intro: first section if it precedes vocal activity (low lyric density)
    2. Outro: last section if vocal activity has ended
    3. Chorus: sections with highest cross-track lyric phrase repetition
    4. Breakdown: lowest-energy section in second half
    5. Double chorus: high-energy section after breakdown
    6. Pre-chorus: rising-energy section before a chorus
    7. Verse: remaining sections with lyric content
    8. Unknown: very short or ambiguous sections

    Args:
        sections: Section dicts.
        energies: Mean energy per section.
        lyric_data: Lyric features per section.
        duration: Total duration.
        short_id: For logging.

    Returns:
        List of SectionLabel objects.
    """
    n = len(sections)
    labels = ["unknown"] * n
    confidences = [LOW_CONFIDENCE] * n
    sources = ["acoustic"] * n

    if n == 0:
        return []

    # --- Step 1: Find cross-section repeated phrases (chorus signal) ---
    # Phrases that appear in multiple sections are likely hook/chorus phrases
    all_phrase_counts: Counter = Counter()
    for ld in lyric_data:
        all_phrase_counts.update(ld["phrases"])

    # A phrase is a "cross-section phrase" if it appears in 2+ sections
    cross_section_phrases: set[str] = set()
    phrase_section_counts: Counter = Counter()
    for ld in lyric_data:
        for phrase in ld["phrases"]:
            if ld["phrases"][phrase] > 0:
                phrase_section_counts[phrase] += 1
    cross_section_phrases = {p for p, c in phrase_section_counts.items() if c >= 2}

    # Repetition score per section: count cross-section phrases
    repetition_scores = []
    for ld in lyric_data:
        score = sum(ld["phrases"][p] for p in cross_section_phrases if p in ld["phrases"])
        repetition_scores.append(score)

    # --- Step 2: Identify intro ---
    # First section is intro if: early in track AND low lyric density
    if n >= 2:
        first = sections[0]
        first_lyric = lyric_data[0]["lyric_density"]
        is_early = first["end"] <= duration * 0.15
        is_quiet_start = energies[0] < (np.mean(energies) * 0.9)

        if first_lyric < 0.1 or (is_early and is_quiet_start):
            labels[0] = "intro"
            conf = HIGH_CONFIDENCE if first_lyric < 0.05 else MEDIUM_CONFIDENCE
            confidences[0] = conf
            sources[0] = "hybrid" if first_lyric < 0.05 else "acoustic"

    # --- Step 3: Identify outro ---
    # Last section is outro if: late in track AND declining energy/lyrics
    if n >= 2:
        last_lyric = lyric_data[-1]["lyric_density"]
        is_late = sections[-1]["start"] >= duration * 0.85
        is_quiet_end = energies[-1] < (np.mean(energies) * 0.9)

        if is_late and (last_lyric < 0.1 or is_quiet_end):
            labels[-1] = "outro"
            conf = MEDIUM_CONFIDENCE
            confidences[-1] = conf
            sources[-1] = "acoustic"

    # --- Step 4: Identify chorus candidates ---
    # Sections with high repetition score and above-average energy
    mean_energy = float(np.mean(energies))
    max_rep = max(repetition_scores) if repetition_scores else 0

    chorus_indices = []
    if max_rep > 0:
        for i in range(n):
            if labels[i] != "unknown":
                continue
            rep_norm = repetition_scores[i] / max_rep
            energy_above_avg = energies[i] >= mean_energy * 0.95

            if rep_norm >= 0.7 and energy_above_avg:
                labels[i] = "chorus"
                # Both lyric and energy signals agree
                confidences[i] = HIGH_CONFIDENCE
                sources[i] = "hybrid"
                chorus_indices.append(i)
            elif rep_norm >= 0.5:
                labels[i] = "chorus"
                confidences[i] = MEDIUM_CONFIDENCE
                sources[i] = "lyric"
                chorus_indices.append(i)

    # --- Step 5: Identify breakdown ---
    # Lowest-energy unlabelled section in the second half
    second_half_start = n // 2
    unlabelled_second_half = [i for i in range(second_half_start, n) if labels[i] == "unknown"]
    breakdown_idx = None
    if unlabelled_second_half:
        min_energy_idx = min(unlabelled_second_half, key=lambda i: energies[i])
        global_mean = float(np.mean(energies))
        if energies[min_energy_idx] < global_mean * 0.7:
            labels[min_energy_idx] = "breakdown"
            confidences[min_energy_idx] = MEDIUM_CONFIDENCE
            sources[min_energy_idx] = "acoustic"
            breakdown_idx = min_energy_idx

    # --- Step 6: Identify double chorus ---
    # High-energy unlabelled section immediately after breakdown
    if breakdown_idx is not None:
        for j in range(breakdown_idx + 1, n):
            if labels[j] == "unknown":
                if energies[j] >= mean_energy * 1.05:
                    labels[j] = "double_chorus"
                    confidences[j] = MEDIUM_CONFIDENCE
                    sources[j] = "acoustic"
                break  # only check the immediate next section

    # --- Step 7: Pre-chorus ---
    # Unlabelled section immediately before a chorus with rising energy
    for i in range(1, n - 1):
        if labels[i] != "unknown":
            continue
        next_is_chorus = i + 1 < n and labels[i + 1] == "chorus"
        energy_rising = energies[i] > energies[max(0, i - 1)] * 1.05
        if next_is_chorus and energy_rising:
            labels[i] = "pre_chorus"
            confidences[i] = LOW_CONFIDENCE
            sources[i] = "acoustic"

    # --- Step 8: Label remaining as verse ---
    for i in range(n):
        if labels[i] == "unknown":
            has_lyrics = lyric_data[i]["lyric_density"] > 0.05
            if has_lyrics:
                labels[i] = "verse"
                confidences[i] = LOW_CONFIDENCE
                sources[i] = "lyric" if lyric_data[i]["lyric_density"] > 0.2 else "acoustic"

    # --- Build result list ---
    result = []
    for i, sec in enumerate(sections):
        result.append(
            SectionLabel(
                position=i,
                start=sec["start"],
                end=sec["end"],
                label=labels[i],
                label_confidence=confidences[i],
                label_source=sources[i],
                mean_energy=energies[i],
                lyric_density=lyric_data[i]["lyric_density"],
                repeated_phrase=lyric_data[i]["repeated_phrase"],
            )
        )

    return result


# ---------------------------------------------------------------------------
# SQLite writes (pass 2)
# ---------------------------------------------------------------------------


def _write_labels(
    db_path: Path,
    track_id: str,
    labels: list[SectionLabel],
    short_id: str,
) -> None:
    """Update section rows with alignment results."""
    conn = get_connection(db_path)
    try:
        with conn:
            for section in labels:
                conn.execute(
                    """
                    UPDATE sections SET
                        label = ?,
                        label_confidence = ?,
                        label_source = ?,
                        mean_energy = ?,
                        lyric_density = ?,
                        repeated_phrase = ?
                    WHERE track_id = ? AND position = ?
                    """,
                    (
                        section.label,
                        round(section.label_confidence, 4),
                        section.label_source,
                        round(section.mean_energy, 4),
                        round(section.lyric_density, 4),
                        section.repeated_phrase,
                        track_id,
                        section.position,
                    ),
                )
    finally:
        conn.close()
