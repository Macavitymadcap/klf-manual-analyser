"""report/queries.py — SQLite queries that feed the report templates."""

import json
from pathlib import Path

from manual_analyser.db import get_connection
from manual_analyser.report.types import (
    CriterionResult,
    CriterionSummaryData,
    SectionData,
    TrackReportData,
    TrackRowData,
    TranscriptLine,
)


def load_track_report_data(
    track_id: str,
    mode: str,
    db_path: Path,
    stem_base_url: str,
) -> TrackReportData:
    """Load everything needed to render one track detail page."""
    conn = get_connection(db_path)
    try:
        track = dict(conn.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,)).fetchone())
        sections = _load_sections(conn, track_id)
        rms_profile = _load_rms_profile(conn, track_id)
        criteria = _load_criteria(conn, track_id, mode)
        transcript = _load_transcript(conn, track_id, sections)
        patterns = _load_beat_patterns(conn, track_id)
        overall, passed, total = _compute_score(criteria)
    finally:
        conn.close()

    return TrackReportData(
        track_id=track_id,
        artist=track.get("artist"),
        song_name=track.get("song_name"),
        filename=track["filename"],
        duration=track.get("duration"),
        bpm=track.get("bpm"),
        key=track.get("key"),
        mode=track.get("mode"),
        groove_feel=track.get("groove_feel"),
        energy_shape=track.get("energy_shape"),
        danceability=track.get("danceability"),
        hook_phrase=track.get("hook_phrase"),
        hook_repetition_count=track.get("hook_repetition_count"),
        hook_first_appearance=track.get("hook_first_appearance"),
        overall_score=overall,
        passed_count=passed,
        total_count=total,
        sections=sections,
        rms_profile=rms_profile,
        criteria=criteria,
        transcript=transcript,
        kick_pattern=patterns.get("kick_pattern"),
        snare_pattern=patterns.get("snare_pattern"),
        hihat_pattern=patterns.get("hihat_pattern"),
        syncopation_score=patterns.get("syncopation_score"),
        rhythmic_density=patterns.get("rhythmic_density"),
        mode_name=mode,
        stem_base_url=stem_base_url,
    )


def load_track_row(track_id: str, mode: str, db_path: Path, detail_url: str) -> TrackRowData:
    """Load lightweight track summary for the ranking table."""
    conn = get_connection(db_path)
    try:
        track = conn.execute("SELECT artist, song_name FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
        overall, passed, total = _compute_score(_load_criteria(conn, track_id, mode))
    finally:
        conn.close()
    return TrackRowData(
        track_id=track_id,
        artist=track["artist"] if track else None,
        song_name=track["song_name"] if track else None,
        overall_score=overall,
        passed_count=passed,
        total_count=total,
        detail_url=detail_url,
    )


def load_criterion_summaries(mode: str, db_path: Path) -> list[CriterionSummaryData]:
    """Aggregate pass rate and mean score per criterion across all tracks."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT criterion_id,
                      AVG(CASE WHEN passed=1 THEN 1.0 ELSE 0.0 END) as pass_rate,
                      AVG(score) as mean_score
               FROM scores
               WHERE mode = ? AND score IS NOT NULL
               GROUP BY criterion_id""",
            (mode,),
        ).fetchall()
    finally:
        conn.close()
    return [
        CriterionSummaryData(
            criterion_id=r["criterion_id"],
            pass_rate=round(r["pass_rate"] or 0.0, 3),
            mean_score=round(r["mean_score"] or 0.0, 3),
        )
        for r in rows
    ]


def load_scored_track_ids(mode: str, db_path: Path) -> list[str]:
    """Return track_ids with scores, ordered by mean score descending."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT track_id, AVG(score) as avg_score
               FROM scores WHERE mode = ? AND score IS NOT NULL
               GROUP BY track_id ORDER BY avg_score DESC""",
            (mode,),
        ).fetchall()
    finally:
        conn.close()
    return [r["track_id"] for r in rows]


def _load_sections(conn, track_id: str) -> list[SectionData]:
    rows = conn.execute(
        """SELECT s.label, s.start, s.end, s.duration, s.mean_energy,
                  s.label_confidence, cp.progression
           FROM sections s
           LEFT JOIN chord_progressions cp ON cp.section_id = s.id
           WHERE s.track_id = ? ORDER BY s.position""",
        (track_id,),
    ).fetchall()
    return [
        SectionData(
            label=r["label"],
            start=r["start"],
            end=r["end"],
            duration=r["duration"],
            mean_energy=r["mean_energy"],
            label_confidence=r["label_confidence"],
            chord_progression=r["progression"],
        )
        for r in rows
    ]


def _load_rms_profile(conn, track_id: str) -> list[float]:
    row = conn.execute("SELECT rms_profile_json FROM tracks_timeseries WHERE track_id = ?", (track_id,)).fetchone()
    if not row:
        return []
    return json.loads(row["rms_profile_json"])


def _load_criteria(conn, track_id: str, mode: str) -> list[CriterionResult]:
    rows = conn.execute(
        "SELECT criterion_id, score, reasoning, passed FROM scores WHERE track_id = ? AND mode = ?",
        (track_id, mode),
    ).fetchall()
    return [
        CriterionResult(
            criterion_id=r["criterion_id"],
            rule="unknown",  # rule type not stored in scores; renderer uses it for display only
            passed=bool(r["passed"]),
            score=r["score"],
            reasoning=r["reasoning"],
            weight=1.0,  # weight not stored in scores; shown as fallback
        )
        for r in rows
    ]


def _load_transcript(conn, track_id: str, sections: list[SectionData]) -> list[TranscriptLine]:
    rows = conn.execute(
        "SELECT start, end, text FROM transcript_segments WHERE track_id = ? ORDER BY start",
        (track_id,),
    ).fetchall()
    return [
        TranscriptLine(
            start=r["start"],
            end=r["end"],
            text=r["text"],
            section_label=_section_label_at(r["start"], sections),
        )
        for r in rows
    ]


def _load_beat_patterns(conn, track_id: str) -> dict:
    row = conn.execute(
        "SELECT kick_pattern, snare_pattern, hihat_pattern, syncopation_score, rhythmic_density"
        " FROM beat_patterns WHERE track_id = ?",
        (track_id,),
    ).fetchone()
    return dict(row) if row else {}


def _section_label_at(time: float, sections: list[SectionData]) -> str | None:
    for s in sections:
        if s.start <= time < s.end:
            return s.label
    return None


def _compute_score(criteria: list[CriterionResult]) -> tuple[float, int, int]:
    scored = [c for c in criteria if c.score is not None]
    if not scored:
        return 0.0, 0, len(criteria)
    overall = round(sum(c.score for c in scored) / len(scored), 3)
    passed = sum(1 for c in scored if c.passed)
    return overall, passed, len(criteria)
