"""aggregation/types.py — Output types from the aggregation stage."""

from dataclasses import dataclass, field


@dataclass
class CriterionSummary:
    """Aggregate stats for one criterion across all scored tracks."""

    criterion_id: str
    pass_rate: float  # 0.0–1.0
    mean_score: float  # 0.0–1.0; excludes null scores
    scored_track_count: int  # tracks with non-null score for this criterion


@dataclass
class TrackSummary:
    """Per-track data for the report track list."""

    track_id: str
    artist: str | None
    song_name: str | None
    overall_score: float
    passed_count: int
    total_count: int


@dataclass
class ClusterInfo:
    """Optional Qdrant cluster data — absent if Qdrant unavailable."""

    cluster_id: int
    track_ids: list[str]
    dominant_features: dict[str, str]  # e.g. {"groove_feel": "straight", "key": "C"}


@dataclass
class AggregateReport:
    """Complete aggregation result — passed to render.py."""

    mode: str
    track_count: int
    criteria: list[CriterionSummary]
    tracks: list[TrackSummary]

    # Modal feature values across all tracks
    modal_bpm: float | None
    modal_key: str | None
    modal_mode: str | None
    modal_groove_feel: str | None
    modal_energy_shape: str | None
    modal_structure: list[str]  # most common section label sequence

    # LLM-generated recipe text (None if generation failed)
    recipe: str | None
    recipe_error: str | None  # error message if recipe failed

    # Optional Qdrant cluster data
    clusters: list[ClusterInfo] = field(default_factory=list)
