"""pipeline/scoring_runner.py — Run Stage 6 scoring for one track."""

import logging
from pathlib import Path

from manual_analyser.pipeline.types import TrackState
from manual_analyser.scoring import llm as llm_module
from manual_analyser.scoring.criteria import evaluate_all_deterministic, load_mode
from manual_analyser.scoring.prompt import build_all_llm_prompts

logger = logging.getLogger(__name__)


def run_scoring(track_id: str, mode: str, db_path: Path, state: TrackState) -> None:
    """
    Evaluate all criteria for one track and persist scores to SQLite.

    Deterministic rules are evaluated immediately.
    LLM rules are batched and sent to Ollama.
    Failures are recorded in state.failed_stages.
    """
    try:
        mode_config = load_mode(mode)
    except Exception as exc:
        logger.error("[%s] [scoring] Could not load mode '%s': %s", state.short_id, mode, exc)
        state.failed_stages.append("scoring")
        return

    _run_deterministic(track_id, mode, mode_config, db_path, state)
    _run_llm(track_id, mode, mode_config, db_path, state)


def _run_deterministic(track_id, mode, mode_config, db_path, state):
    try:
        evaluate_all_deterministic(mode_config, track_id, db_path)
        logger.info("[%s] [scoring] Deterministic criteria done", state.short_id)
    except Exception as exc:
        logger.error("[%s] [scoring] Deterministic scoring failed: %s", state.short_id, exc)
        state.failed_stages.append("scoring_deterministic")


def _run_llm(track_id, mode, mode_config, db_path, state):
    try:
        packages = build_all_llm_prompts(mode_config, track_id, db_path)
        llm_module.score_all(packages, track_id, mode, db_path)
        logger.info("[%s] [scoring] LLM criteria done (%d)", state.short_id, len(packages))
    except Exception as exc:
        logger.error("[%s] [scoring] LLM scoring failed: %s", state.short_id, exc)
        state.failed_stages.append("scoring_llm")
