"""
scoring/prompt.py — LLM prompt construction for scoring.

Responsibilities:
  - Fetch all field values needed for an LLM criterion from SQLite
  - Assemble a structured user prompt containing:
      - Criterion name and description
      - Actual field values from the database (labelled, human-readable)
      - The criterion's prompt_hint verbatim
      - Null-field caveats when data is missing
  - Return a PromptPackage ready for llm.py to send to Ollama

The system prompt comes from the ModeConfig (loaded from TOML).
The user prompt is constructed per-criterion here.

This module does NOT make HTTP calls. That is llm.py's job.
This module does NOT write to SQLite.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from manual_analyser.db import get_connection
from manual_analyser.scoring.criteria import Criterion, ModeConfig

logger = logging.getLogger(__name__)

# Fields that need human-readable descriptions for the LLM
_FIELD_DESCRIPTIONS: dict[str, str] = {
    # Tempo
    "tracks.bpm": "BPM (beats per minute)",
    "tracks.bpm_confidence": "BPM detection confidence (0.0–1.0)",
    "tracks.time_signature": "Time signature (3 or 4)",
    "tracks.tempo_stability": "Tempo stability (0.0–1.0; 1.0 = perfectly metronomic)",
    # Groove
    "tracks.danceability": "Danceability (0.0–1.0)",
    "tracks.self_similarity_score": "Self-similarity score (0.0–1.0; high = consistent groove)",
    "tracks.beat_regularity": "Beat regularity (0.0–1.0; 1.0 = metronomic)",
    "tracks.groove_consistency": "Groove consistency composite (0.0–1.0)",
    "tracks.repetition_score": "Repetition score (0.0–1.0; high = track recurs)",
    # Rhythm
    "tracks.groove_feel": "Groove feel ('straight', 'swung', or 'unclear')",
    # Harmony
    "tracks.key": "Musical key (e.g. 'C', 'F#')",
    "tracks.mode": "Mode ('major' or 'minor')",
    "tracks.key_confidence": "Key detection confidence (0.0–1.0)",
    # Energy
    "tracks.loudness_db": "Integrated loudness (normalised 0.0–1.0)",
    "tracks.dynamic_range_db": "Dynamic range (normalised 0.0–1.0)",
    "tracks.verse_chorus_delta": "Verse-to-chorus energy lift (normalised; 0.15 ≈ 3dB)",
    "tracks.energy_shape": "Energy shape ('building', 'flat', 'peaked', or 'unclear')",
    # Lyrics / hook
    "tracks.unique_word_ratio": "Unique word ratio (0.0–1.0; low = more repetitive)",
    "tracks.hook_repetition_count": "Hook phrase repetition count",
    "tracks.hook_first_appearance": "Hook first appearance (seconds into track)",
    "tracks.hook_phrase": "Most repeated phrase (hook)",
    "tracks.song_name": "Song title (from filename)",
    "tracks.artist": "Artist name (from filename)",
    # Sections
    "sections.label": "Section labels (ordered sequence)",
    "sections.label_confidence": "Section label confidence scores (0.0–1.0)",
    "sections.duration": "Section duration (seconds)",
    # Beat patterns
    "beat_patterns.kick_pattern": "Kick drum pattern (16-step binary grid)",
    "beat_patterns.snare_pattern": "Snare pattern (16-step binary grid)",
    "beat_patterns.hihat_pattern": "Hi-hat pattern (16-step binary grid)",
    "beat_patterns.syncopation_score": "Syncopation score (0.0–1.0)",
    "beat_patterns.rhythmic_density": "Rhythmic density (0.0–1.0)",
    # Chord progressions
    "chord_progressions.progression": "Chord progressions per section",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PromptPackage:
    """A complete prompt ready to send to the LLM."""

    criterion_id: str
    system_prompt: str
    user_prompt: str
    has_null_fields: bool  # True if any field values are null
    null_field_names: list[str]  # Names of null fields for logging


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompt(
    criterion: Criterion,
    mode_config: ModeConfig,
    track_id: str,
    db_path: Path,
) -> PromptPackage:
    """
    Build a complete prompt package for an LLM criterion.

    Fetches all field values from SQLite and assembles the user prompt.
    The system prompt comes from the ModeConfig.

    Args:
        criterion: An llm-rule Criterion.
        mode_config: Loaded mode configuration (provides system prompt).
        track_id: Track to score.
        db_path: SQLite database path.

    Returns:
        PromptPackage ready for llm.py.

    Raises:
        ValueError: if criterion.rule is not "llm".
    """
    if not criterion.is_llm:
        raise ValueError(f"build_prompt called on non-llm criterion '{criterion.id}' (rule='{criterion.rule}')")

    # Fetch field values from DB
    field_values = _fetch_field_values(criterion, track_id, db_path)
    null_fields = [k for k, v in field_values.items() if v is None]

    # Build user prompt
    user_prompt = _assemble_user_prompt(criterion, field_values, null_fields)

    return PromptPackage(
        criterion_id=criterion.id,
        system_prompt=mode_config.system_prompt,
        user_prompt=user_prompt,
        has_null_fields=bool(null_fields),
        null_field_names=null_fields,
    )


def build_all_llm_prompts(
    mode_config: ModeConfig,
    track_id: str,
    db_path: Path,
) -> list[PromptPackage]:
    """
    Build prompt packages for all LLM criteria in a mode.

    Args:
        mode_config: Loaded mode configuration.
        track_id: Track to score.
        db_path: SQLite database path.

    Returns:
        List of PromptPackage, one per llm criterion.
    """
    packages = []
    for criterion in mode_config.criteria:
        if criterion.is_llm:
            try:
                package = build_prompt(criterion, mode_config, track_id, db_path)
                packages.append(package)
            except Exception as e:
                logger.exception("Failed to build prompt for criterion '%s': %s", criterion.id, e, exc_info=True)
    return packages


# ---------------------------------------------------------------------------
# Field value fetching
# ---------------------------------------------------------------------------


def _fetch_field_values(
    criterion: Criterion,
    track_id: str,
    db_path: Path,
) -> dict[str, object]:
    """
    Fetch all field values needed for this criterion from SQLite.

    Handles tracks.*, sections.*, beat_patterns.*, chord_progressions.*
    field references. Returns a dict keyed by field name.

    Args:
        criterion: Criterion with db_field or db_fields.
        track_id: Track to query.
        db_path: SQLite database path.

    Returns:
        Dict mapping field name → value (or None if null/missing).
    """
    fields = criterion.fields
    values: dict[str, object] = {}

    conn = get_connection(db_path)
    try:
        # Batch fields by table for efficient querying
        tracks_fields = [f for f in fields if f.startswith("tracks.")]
        sections_fields = [f for f in fields if f.startswith("sections.")]
        beat_fields = [f for f in fields if f.startswith("beat_patterns.")]
        chord_fields = [f for f in fields if f.startswith("chord_progressions.")]

        # Fetch tracks fields
        if tracks_fields:
            cols = [f.split(".", 1)[1] for f in tracks_fields]
            row = conn.execute(
                f"SELECT {', '.join(cols)} FROM tracks WHERE track_id = ?",
                (track_id,),
            ).fetchone()
            for field, col in zip(tracks_fields, cols):
                values[field] = row[col] if row else None

        if sections_fields:
            values = _fetch_sections(conn, track_id, sections_fields, values)

        if beat_fields:
            values = _fetch_beat_patterns(conn, track_id, beat_fields, values)

        if chord_fields:
            values = _fetch_chord_progressions(conn, track_id, chord_fields, values)

    finally:
        conn.close()

    return values


def _fetch_sections(conn, track_id: str, sections_fields: list[str], values: dict[str, object]) -> dict[str, object]:
    section_rows = conn.execute(
        "SELECT position, label, label_confidence, start, end, duration "
        "FROM sections WHERE track_id = ? ORDER BY position",
        (track_id,),
    ).fetchall()

    for field in sections_fields:
        col = field.split(".", 1)[1]
        if col == "label":
            # Return ordered sequence with confidence
            if section_rows:
                seq = [
                    f"{r['label']} (pos={r['position']}, "
                    f"conf={r['label_confidence']:.2f}, "
                    f"{r['start']:.1f}s–{r['end']:.1f}s)"
                    for r in section_rows
                ]
                values[field] = " → ".join(seq)
            else:
                values[field] = None
        elif col == "duration":
            # Return duration of intro section specifically (for intro_length)
            intro_row = next((r for r in section_rows if r["label"] == "intro"), None)
            values[field] = float(intro_row["duration"]) if intro_row else None
        else:
            values[field] = None  # other section fields not commonly used

    return values


def _fetch_beat_patterns(conn, track_id: str, beat_fields: list[str], values: dict[str, object]) -> dict[str, object]:
    bp_row = conn.execute(
        "SELECT kick_pattern, snare_pattern, hihat_pattern, "
        "syncopation_score, rhythmic_density "
        "FROM beat_patterns WHERE track_id = ?",
        (track_id,),
    ).fetchone()
    for field in beat_fields:
        col = field.split(".", 1)[1]
        values[field] = bp_row[col] if bp_row else None

    return values


def _fetch_chord_progressions(
    conn, track_id: str, chord_fields: list[str], values: dict[str, object]
) -> dict[str, object]:
    cp_rows = conn.execute(
        """
      SELECT s.position, cp.progression
      FROM chord_progressions cp
      JOIN sections s ON cp.section_id = s.id
      WHERE s.track_id = ?
      ORDER BY s.position
      """,
        (track_id,),
    ).fetchall()
    for field in chord_fields:
        col = field.split(".", 1)[1]
        if col == "progression" and cp_rows:
            progs = [f"Section {r['position']}: {r['progression']}" for r in cp_rows]
            values[field] = "; ".join(progs)
        else:
            values[field] = None

    return values


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _assemble_user_prompt(
    criterion: Criterion,
    field_values: dict[str, object],
    null_fields: list[str],
) -> str:
    """
    Assemble the user prompt string for an LLM criterion.

    Format:
      CRITERION: {name}
      {description}

      FIELD VALUES:
      - {field_description}: {value}
      ...

      [NULL FIELDS NOTE if any]

      SCORING GUIDANCE:
      {prompt_hint}

    Args:
        criterion: The criterion being scored.
        field_values: Dict of field → value.
        null_fields: List of field names with null values.

    Returns:
        User prompt string.
    """
    lines: list[str] = []

    # Header
    lines.append(f"CRITERION: {criterion.name}")
    lines.append("")
    lines.append(criterion.description.strip())
    lines.append("")

    # Field values
    lines.append("FIELD VALUES:")
    for field, value in field_values.items():
        description = _FIELD_DESCRIPTIONS.get(field, field)
        if value is None:
            lines.append(f"  - {description}: [not available]")
        else:
            lines.append(f"  - {description}: {_format_value(field, value)}")

    # Null field caveat
    if null_fields:
        lines.append("")
        lines.append("NOTE: The following fields have no data (analysis stage may not")
        lines.append("have run, or the feature was not detected). Score based on")
        lines.append("available data only; acknowledge missing data in your reasoning.")
        for f in null_fields:
            desc = _FIELD_DESCRIPTIONS.get(f, f)
            lines.append(f"  - {desc}")

    # Scoring guidance
    lines.append("")
    lines.append("SCORING GUIDANCE:")
    lines.append(criterion.prompt_hint.strip() if criterion.prompt_hint else "")

    return "\n".join(lines)


def _format_value(field: str, value: object) -> str:
    """
    Format a field value for human-readable prompt output.

    Applies field-specific formatting:
    - Floats rounded to 3 decimal places
    - Beat patterns rendered with visual spacing
    - Long strings truncated

    Args:
        field: The field name (e.g. "tracks.bpm").
        value: The raw value.

    Returns:
        Formatted string.
    """
    if isinstance(value, float):
        return f"{value:.3f}"

    if isinstance(value, str):
        # Beat patterns — add spaces for readability
        if "pattern" in field and len(value) == 16:
            # Group into 4 beats: "1000 1000 1000 1000"
            return " ".join(value[i : i + 4] for i in range(0, 16, 4))
        # Truncate very long strings
        if len(value) > 500:
            return value[:497] + "..."

    return str(value)
