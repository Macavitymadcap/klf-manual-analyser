"""scoring/llm/types.py — Result types for LLM scoring."""

from dataclasses import dataclass


@dataclass
class LlmScore:
    """A parsed, validated score returned from Ollama."""

    criterion_id: str
    score: float  # normalised 0.0–1.0
    raw_score: int  # original 0–10 from LLM
    reasoning: str
    passed: bool  # score >= 0.5


@dataclass
class LlmFailure:
    """Represents a failed LLM call — score=null, reasoning records why."""

    criterion_id: str
    reason: str  # "parse_error" | "timeout" | "http_error"


# Union type for callers
LlmResult = LlmScore | LlmFailure
