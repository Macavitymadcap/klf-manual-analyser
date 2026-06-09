"""report/types.py — Data types passed to Jinja2 templates."""

from dataclasses import dataclass


@dataclass
class SectionData:
    label: str
    start: float
    end: float
    duration: float
    mean_energy: float | None
    label_confidence: float
    chord_progression: str | None


@dataclass
class CriterionResult:
    criterion_id: str
    rule: str  # "lte" | "gte" | "range" | "exists" | "llm"
    passed: bool
    score: float | None  # None if LLM failed
    reasoning: str | None
    weight: float


@dataclass
class TranscriptLine:
    start: float
    end: float
    text: str
    section_label: str | None


@dataclass
class TrackReportData:
    """All data needed to render one track detail page."""

    track_id: str
    artist: str | None
    song_name: str | None
    filename: str
    duration: float | None
    bpm: float | None
    key: str | None
    mode: str | None
    groove_feel: str | None
    energy_shape: str | None
    danceability: float | None
    hook_phrase: str | None
    hook_repetition_count: int | None
    hook_first_appearance: float | None

    overall_score: float
    passed_count: int
    total_count: int

    sections: list[SectionData]
    rms_profile: list[float]  # for energy chart
    criteria: list[CriterionResult]
    transcript: list[TranscriptLine]

    kick_pattern: str | None
    snare_pattern: str | None
    hihat_pattern: str | None
    syncopation_score: float | None
    rhythmic_density: float | None

    mode_name: str  # "1988" | "contemporary" | "1920s_1930s"
    stem_base_url: str  # e.g. "/stems/{track_id}"


@dataclass
class CriterionSummaryData:
    criterion_id: str
    pass_rate: float
    mean_score: float


@dataclass
class TrackRowData:
    """Lightweight summary for the ranking table in summary.html."""

    track_id: str
    artist: str | None
    song_name: str | None
    overall_score: float
    passed_count: int
    total_count: int
    detail_url: str  # relative path to track HTML file


@dataclass
class SummaryReportData:
    """All data needed to render summary.html."""

    mode_name: str
    track_count: int
    rendered_at: str  # ISO 8601

    recipe: str | None
    recipe_error: str | None

    modal_bpm: float | None
    modal_key: str | None
    modal_mode: str | None
    modal_groove_feel: str | None
    modal_energy_shape: str | None
    modal_structure: list[str]

    overall_pass_rate: float
    criteria: list[CriterionSummaryData]
    tracks: list[TrackRowData]
