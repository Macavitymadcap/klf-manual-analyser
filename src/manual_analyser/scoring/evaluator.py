"""
scoring/evaluator.py — Deterministic criterion evaluation and score aggregation.

This module does NOT make LLM calls and does NOT write to SQLite.

Public API:
    evaluate_deterministic(criterion, track_id, db_path) -> EvaluationResult
    evaluate_all_deterministic(mode_config, track_id, db_path) -> dict[str, EvaluationResult]
    make_llm_placeholder(criterion) -> EvaluationResult
    compute_overall_score(results, mode_config) -> float
"""

import logging
from pathlib import Path

from manual_analyser.db import get_connection
from manual_analyser.scoring.types import (
    Criterion,
    EvaluationResult,
    ModeConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_deterministic(
    criterion: Criterion,
    track_id: str,
    db_path: Path,
) -> EvaluationResult:
    """
    Evaluate a deterministic criterion against a track's DB values.

    Only valid for non-llm rules. Raises ValueError if called on an
    llm criterion.

    Args:
        criterion: The criterion to evaluate.
        track_id: Track to evaluate against.
        db_path: Path to SQLite database.

    Returns:
        EvaluationResult with score, passed, and raw_value set.
        score is 1.0 for pass, 0.0 for fail (proportional for range).
        needs_llm is always False.
    """
    if criterion.is_llm:
        raise ValueError(f"evaluate_deterministic called on llm criterion '{criterion.id}'")

    raw_value = _fetch_field_value(criterion, track_id, db_path)

    if raw_value is None:
        return EvaluationResult(
            criterion_id=criterion.id,
            rule=criterion.rule,
            passed=False,
            score=0.0,
            raw_value=None,
            reasoning="Field value is null — analysis stage may not have run.",
            needs_llm=False,
        )

    passed, score = _apply_rule(criterion, raw_value)

    return EvaluationResult(
        criterion_id=criterion.id,
        rule=criterion.rule,
        passed=passed,
        score=score,
        raw_value=raw_value,
        reasoning=criterion.fail_message if not passed else None,
        needs_llm=False,
    )


def make_llm_placeholder(criterion: Criterion) -> EvaluationResult:
    """
    Return a placeholder EvaluationResult for an LLM criterion.

    needs_llm=True signals the scoring orchestrator to replace this
    after the LLM call.
    """
    return EvaluationResult(
        criterion_id=criterion.id,
        rule=criterion.rule,
        passed=False,  # placeholder — updated by llm.py
        score=0.0,  # placeholder — updated by llm.py
        raw_value=None,
        reasoning=None,  # filled by llm.py
        needs_llm=True,
    )


def evaluate_all_deterministic(
    mode_config: ModeConfig,
    track_id: str,
    db_path: Path,
) -> dict[str, EvaluationResult]:
    """
    Evaluate all deterministic criteria for a track.

    LLM criteria are included as placeholders (needs_llm=True).

    Args:
        mode_config: Loaded mode configuration.
        track_id: Track to evaluate.
        db_path: SQLite database path.

    Returns:
        Dict of criterion_id → EvaluationResult.
    """
    results: dict[str, EvaluationResult] = {}

    for criterion in mode_config.criteria:
        if criterion.is_deterministic:
            try:
                results[criterion.id] = evaluate_deterministic(criterion, track_id, db_path)
            except Exception as e:
                logger.exception(
                    "Failed to evaluate criterion '%s': %s",
                    criterion.id,
                    e,
                    exc_info=True,
                )
                results[criterion.id] = EvaluationResult(
                    criterion_id=criterion.id,
                    rule=criterion.rule,
                    passed=False,
                    score=0.0,
                    raw_value=None,
                    reasoning=f"Evaluation error: {e}",
                    needs_llm=False,
                )
        else:
            results[criterion.id] = make_llm_placeholder(criterion)

    return results


def compute_overall_score(
    results: dict[str, EvaluationResult],
    mode_config: ModeConfig,
) -> float:
    """
    Compute the weighted overall compliance score from criterion results.

    Only includes criteria that have been fully scored (needs_llm=False).
    LLM placeholders that haven't been filled are excluded.

    Args:
        results: Dict of criterion_id → EvaluationResult.
        mode_config: Mode config for weights.

    Returns:
        Weighted score 0.0–1.0.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for criterion in mode_config.criteria:
        result = results.get(criterion.id)
        if result is None or result.needs_llm:
            continue
        total_weight += criterion.weight
        weighted_sum += result.score * criterion.weight

    if total_weight == 0:
        return 0.0

    return float(weighted_sum / total_weight)


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def _apply_rule(criterion: Criterion, raw_value: float) -> tuple[bool, float]:
    """
    Apply a deterministic rule and return (passed, score).

    Score is 1.0 for pass, 0.0 for fail, except range which gives a
    proportional score based on distance from the nearest bound.
    """
    rule = criterion.rule

    if rule == "lte":
        return _less_than_equal(raw_value, criterion.threshold)
    elif rule == "gte":
        return _greater_than_equal(raw_value, criterion.threshold)
    elif rule == "eq":
        return _equal(raw_value, criterion.threshold)
    elif rule == "range":
        return _range(raw_value, criterion.threshold_min, criterion.threshold_max)
    elif rule == "exists":
        return _exists(raw_value)

    return False, 0.0


def _less_than_equal(raw_value: float, threshold: float) -> tuple[bool, float]:
    passed = raw_value <= threshold
    return passed, 1.0 if passed else 0.0


def _greater_than_equal(raw_value: float, threshold: float) -> tuple[bool, float]:
    passed = raw_value >= threshold
    return passed, 1.0 if passed else 0.0


def _equal(raw_value: float, threshold: float) -> tuple[bool, float]:
    passed = abs(raw_value - threshold) < 1e-9
    return passed, 1.0 if passed else 0.0


def _range(raw_value: float, threshold_min: float, threshold_max: float) -> tuple[bool, float]:
    passed = threshold_min <= raw_value <= threshold_max
    if passed:
        return True, 1.0
    # Proportional score: distance from nearest bound
    if raw_value < threshold_min:
        dist = (threshold_min - raw_value) / (threshold_min + 1e-9)
    else:
        dist = (raw_value - threshold_max) / (threshold_max + 1e-9)
    return False, float(max(0.0, 1.0 - dist))


def _exists(raw_value: float) -> tuple[bool, float]:
    # raw_value for exists is 1.0 (found) or 0.0 (not found)
    passed = raw_value >= 1.0
    return passed, 1.0 if passed else 0.0


# ---------------------------------------------------------------------------
# DB value fetching
# ---------------------------------------------------------------------------


def _fetch_field_value(
    criterion: Criterion,
    track_id: str,
    db_path: Path,
) -> float | None:
    """
    Fetch the scalar value for a deterministic criterion from SQLite.

    Handles:
      - tracks.*              — simple column lookup
      - sections.label (exists) — returns 1.0 if matching label exists
      - sections.<col> WHERE label=... — e.g. intro duration

    Returns float value, or None if null / not found.
    """
    db_field = criterion.db_field

    conn = get_connection(db_path)
    try:
        if criterion.rule == "exists":
            count = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE track_id = ? AND label = ?",
                (track_id, criterion.value),
            ).fetchone()[0]
            return float(count)

        if "." not in db_field:
            logger.warning("db_field '%s' has no table prefix", db_field)
            return None

        table, column = db_field.split(".", 1)

        # Handle "sections.duration WHERE label='intro'" style references
        where_clause = None
        if " WHERE " in column:
            column, where_part = column.split(" WHERE ", 1)
            where_clause = where_part.strip()

        if table == "tracks":
            row = conn.execute(
                f"SELECT {column} FROM tracks WHERE track_id = ?",
                (track_id,),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            return float(row[0])

        elif table == "sections":
            if where_clause:
                query = f"SELECT {column} FROM sections WHERE track_id = ? AND {where_clause} ORDER BY position LIMIT 1"
                row = conn.execute(query, (track_id,)).fetchone()
            else:
                row = conn.execute(
                    f"SELECT {column} FROM sections WHERE track_id = ? LIMIT 1",
                    (track_id,),
                ).fetchone()

            if row is None or row[0] is None:
                return None
            return float(row[0])

        else:
            logger.warning("Unknown table '%s' in db_field '%s'", table, db_field)
            return None

    finally:
        conn.close()
