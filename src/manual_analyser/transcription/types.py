from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """A single Whisper output segment."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Full transcription result for a track."""

    segments: list[TranscriptSegment]
    full_text: str
    language: str
    hook_phrase: str | None
    hook_repetition_count: int
    hook_first_appearance: float | None  # seconds
    unique_word_ratio: float  # 0.0–1.0
