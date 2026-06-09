from anyio import Path

from manual_analyser.db.connection import get_connection
from manual_analyser.scoring.types import Criterion


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
