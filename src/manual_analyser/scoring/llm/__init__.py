"""scoring/llm/__init__.py — LLM scoring via Ollama.

Public API:
    score_criterion(package, track_id, mode, db_path, model) -> LlmResult
    score_all(packages, track_id, mode, db_path, model) -> list[LlmResult]
    check_ollama(model) -> None   (raises OllamaUnavailableError on failure)
    OllamaUnavailableError
"""

import logging
from pathlib import Path

from manual_analyser.scoring.prompt import PromptPackage

from . import ollama_client
from .constants import DEFAULT_MODEL
from .ollama_client import OllamaTimeoutError, OllamaUnavailableError, check_ollama
from .response_parser import parse, strict_prompt
from .score_writer import write_result
from .types import LlmFailure, LlmResult, LlmScore

logger = logging.getLogger(__name__)

__all__ = [
    "score_criterion",
    "score_all",
    "check_ollama",
    "OllamaUnavailableError",
    "LlmScore",
    "LlmFailure",
    "LlmResult",
]


def score_criterion(
    package: PromptPackage,
    track_id: str,
    mode: str,
    db_path: Path,
    model: str = DEFAULT_MODEL,
) -> LlmResult:
    short = track_id[:8]
    result = _call_with_retry(package, model, short)
    write_result(track_id, mode, result, db_path)
    _log_result(short, result)
    return result


def score_all(
    packages: list[PromptPackage],
    track_id: str,
    mode: str,
    db_path: Path,
    model: str = DEFAULT_MODEL,
) -> list[LlmResult]:
    return [score_criterion(p, track_id, mode, db_path, model) for p in packages]


def _call_with_retry(package: PromptPackage, model: str, short: str) -> LlmResult:
    try:
        raw = ollama_client.chat(package.system_prompt, package.user_prompt, model)
        return _try_parse(package.criterion_id, raw)
    except _ParseError:
        return _retry_strict(package, model, short)
    except OllamaTimeoutError:
        return _retry_timeout(package, model, short)
    except Exception as exc:
        logger.exception("[%s] [scoring] HTTP error on '%s': %s", short, package.criterion_id, exc)
        return LlmFailure(criterion_id=package.criterion_id, reason="http_error")


def _retry_strict(package: PromptPackage, model: str, short: str) -> LlmResult:
    logger.warning("[%s] [scoring] '%s' parse failed, retrying with strict prompt", short, package.criterion_id)
    try:
        raw = ollama_client.chat(package.system_prompt, strict_prompt(package.user_prompt), model)
        return _try_parse(package.criterion_id, raw)
    except _ParseError:
        logger.error("[%s] [scoring] '%s' parse failed on retry", short, package.criterion_id)
        return LlmFailure(criterion_id=package.criterion_id, reason="parse_error")
    except Exception as exc:
        logger.exception("[%s] [scoring] '%s' retry failed: %s", short, package.criterion_id, exc)
        return LlmFailure(criterion_id=package.criterion_id, reason="parse_error")


def _retry_timeout(package: PromptPackage, model: str, short: str) -> LlmResult:
    logger.warning("[%s] [scoring] '%s' timed out, retrying", short, package.criterion_id)
    try:
        raw = ollama_client.chat(package.system_prompt, package.user_prompt, model)
        return _try_parse(package.criterion_id, raw)
    except Exception:
        logger.error("[%s] [scoring] '%s' timed out on retry", short, package.criterion_id)
        return LlmFailure(criterion_id=package.criterion_id, reason="timeout")


def _try_parse(criterion_id: str, raw: str) -> LlmResult:
    try:
        return parse(criterion_id, raw)
    except (ValueError, KeyError) as exc:
        raise _ParseError(str(exc)) from exc


def _log_result(short: str, result: LlmResult) -> None:
    if isinstance(result, LlmFailure):
        logger.warning("[%s] [scoring] '%s' failed: %s", short, result.criterion_id, result.reason)
    else:
        logger.info(
            "[%s] [scoring] '%s' score=%d/10 (%.2f) passed=%s",
            short,
            result.criterion_id,
            result.raw_score,
            result.score,
            result.passed,
        )


class _ParseError(Exception):
    """Internal signal: LLM response could not be parsed."""
