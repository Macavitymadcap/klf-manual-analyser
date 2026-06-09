"""aggregation/recipe.py — LLM call to synthesise the aggregate recipe text."""

import logging

import httpx

from manual_analyser.aggregation.types import AggregateReport

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL = "qwen2.5:14b"
_TIMEOUT = 60.0

_SYSTEM_PROMPT = (
    "You are writing in the voice of The KLF's Bill Drummond and Jimmy Cauty, "
    "authors of The Manual (How To Have A Number One The Easy Way), 1988. "
    "Write with anarchic confidence, imperative sentences, and deliberate provocation. "
    "No hedging. No qualifications. Speak directly to someone who wants a number one."
)


def generate_recipe(report: AggregateReport) -> str:
    """
    Call Ollama to generate a recipe proclamation from aggregate stats.

    Returns the recipe text on success.
    Raises RecipeError on failure — caller writes error to report.
    """
    prompt = _build_prompt(report)
    return _call_ollama(prompt)


def _build_prompt(report: AggregateReport) -> str:
    lines = [
        f"Mode: {report.mode}",
        f"Tracks analysed: {report.track_count}",
        f"Modal BPM: {report.modal_bpm or 'unknown'}",
        f"Modal key: {report.modal_key or 'unknown'} {report.modal_mode or ''}".strip(),
        f"Modal groove: {report.modal_groove_feel or 'unknown'}",
        f"Modal energy shape: {report.modal_energy_shape or 'unknown'}",
        f"Modal structure: {' → '.join(report.modal_structure) if report.modal_structure else 'unknown'}",
        "",
        "Criterion pass rates:",
    ]
    for c in sorted(report.criteria, key=lambda x: x.pass_rate, reverse=True):
        lines.append(f"  {c.criterion_id}: {c.pass_rate:.0%} pass rate (mean score {c.mean_score:.2f})")
    lines += [
        "",
        "Write a recipe — a short, punchy proclamation in The Manual's voice — "
        "describing exactly what a song needs to do to match this set of tracks. "
        "Maximum 200 words. Imperatives only. Make it specific to these numbers.",
    ]
    return "\n".join(lines)


def _call_ollama(prompt: str) -> str:
    resp = httpx.post(
        _OLLAMA_URL,
        json={
            "model": _MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


class RecipeError(Exception):
    """Raised when recipe generation fails. Report renders without recipe."""
