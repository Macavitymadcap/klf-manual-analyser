"""pipeline/types.py — Track status model and run result types."""

from dataclasses import dataclass, field
from enum import Enum


class TrackStatus(str, Enum):
    PENDING = "pending"
    DECODING = "decoding"
    SEPARATING = "separating"
    ANALYSING = "analysing"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    EMBEDDING = "embedding"
    SCORING = "scoring"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class TrackState:
    """In-memory state for one track during a pipeline run."""

    track_id: str
    mp3_path: str
    status: TrackStatus = TrackStatus.PENDING
    failed_stages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def short_id(self) -> str:
        return self.track_id[:8]


@dataclass
class RunSummary:
    """End-of-run summary emitted to the CLI."""

    complete: list[TrackState] = field(default_factory=list)
    partial: list[TrackState] = field(default_factory=list)
    skipped: list[TrackState] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.complete) + len(self.partial) + len(self.skipped)
