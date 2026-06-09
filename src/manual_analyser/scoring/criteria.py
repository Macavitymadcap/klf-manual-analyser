"""
scoring/criteria.py — Public API re-export.

All existing imports continue to work without change:

    from manual_analyser.scoring.criteria import (
        Criterion, ModeConfig, EvaluationResult,
        load_mode, evaluate_deterministic, evaluate_all_deterministic,
        make_llm_placeholder, compute_overall_score,
    )

Implementation is split across:
    scoring/types.py     — Criterion, ModeConfig, EvaluationResult dataclasses
    scoring/loader.py    — TOML loading and validation
    scoring/evaluator.py — deterministic rule evaluation and score aggregation
"""

from manual_analyser.scoring.evaluator import (
    _apply_rule,
    _fetch_field_value,
    compute_overall_score,
    evaluate_all_deterministic,
    evaluate_deterministic,
    make_llm_placeholder,
)
from manual_analyser.scoring.loader import DEFAULT_CONFIG_DIR, _parse_criterion, load_mode
from manual_analyser.scoring.types import (
    ALL_RULES,
    DETERMINISTIC_RULES,
    LLM_RULES,
    Criterion,
    EvaluationResult,
    ModeConfig,
)

__all__ = [
    # Types
    "Criterion",
    "ModeConfig",
    "EvaluationResult",
    # Rule sets
    "ALL_RULES",
    "DETERMINISTIC_RULES",
    "LLM_RULES",
    # Loader
    "load_mode",
    "DEFAULT_CONFIG_DIR",
    "_parse_criterion",
    # Evaluator
    "evaluate_deterministic",
    "evaluate_all_deterministic",
    "make_llm_placeholder",
    "compute_overall_score",
    "_apply_rule",
    "_fetch_field_value",
]
