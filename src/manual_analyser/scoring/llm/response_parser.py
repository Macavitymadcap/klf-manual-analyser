"""scoring/llm/response_parser.py — Parse and validate Ollama JSON responses."""

import json
import logging

from manual_analyser.scoring.llm.constants import PASS_THRESHOLD, SCORE_MAX, SCORE_MIN
from manual_analyser.scoring.llm.types import LlmScore

logger = logging.getLogger(__name__)

_STRICT_SUFFIX = '\n\nRespond with valid JSON only: {"score": <integer 0-10>, "reasoning": "<string>"}'


def strict_prompt(user_prompt: str) -> str:
    """Append a stricter JSON instruction to a user prompt for retry."""
    return user_prompt + _STRICT_SUFFIX


def parse(criterion_id: str, raw_text: str) -> LlmScore:
    """
    Parse and validate a raw Ollama response into an LlmScore.

    Raises:
        ValueError: if the JSON is missing, malformed, or has wrong types.
    """
    data = _extract_json(raw_text)
    raw_score = _extract_score(data)
    reasoning = _extract_reasoning(data)
    clamped = _clamp_score(criterion_id, raw_score)
    return LlmScore(
        criterion_id=criterion_id,
        score=clamped / SCORE_MAX,
        raw_score=clamped,
        reasoning=reasoning,
        passed=clamped >= PASS_THRESHOLD,
    )


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _extract_score(data: dict) -> int:
    score = data.get("score")
    if score is None:
        raise ValueError("Response missing 'score' field")
    return int(score)


def _extract_reasoning(data: dict) -> str:
    return str(data.get("reasoning", ""))


def _clamp_score(criterion_id: str, raw: int) -> int:
    if raw < SCORE_MIN or raw > SCORE_MAX:
        logger.warning("[scoring] '%s' score %d out of range, clamping", criterion_id, raw)
    return max(SCORE_MIN, min(SCORE_MAX, raw))
