"""
scoring/criteria.py — Criteria configuration loader and rule evaluator.

Responsibilities:
  - Load and validate criteria TOML config files
  - Expose a typed Criterion dataclass for use by prompt.py and llm.py
  - Evaluate deterministic rules (lte, gte, eq, range, exists) directly
    from SQLite without any LLM call
  - Return structured EvaluationResult for each criterion

This module does NOT make LLM calls. That is prompt.py + llm.py's job.
This module does NOT write to SQLite. That is the scoring orchestrator's job.

Rule types:
  lte    — pass if db_field <= threshold
  gte    — pass if db_field >= threshold
  eq     — pass if db_field == threshold
  range  — pass if threshold_min <= db_field <= threshold_max
  exists — pass if any sections row has label == value for this track
  llm    — routed to LLM; this module returns None score for these

Validation rules enforced at load time:
  - db_field and db_fields are mutually exclusive
  - threshold rules require db_field (singular)
  - exists rule requires db_field and value
  - llm rule requires db_field or db_fields and prompt_hint
  - weight must be > 0
  - id must be unique within a mode
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from manual_analyser.db import get_connection

logger = logging.getLogger(__name__)

# Valid rule types
DETERMINISTIC_RULES = frozenset({"lte", "gte", "eq", "range", "exists"})
LLM_RULES = frozenset({"llm"})
ALL_RULES = DETERMINISTIC_RULES | LLM_RULES

# Config directory relative to project root
DEFAULT_CONFIG_DIR = Path("config")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Criterion:
    """A single scoring criterion loaded from TOML."""

    id: str
    name: str
    description: str
    weight: float
    rule: str  # "lte" | "gte" | "eq" | "range" | "exists" | "llm"

    # Field mapping — exactly one of db_field or db_fields must be set
    db_field: str | None = None  # single field, e.g. "tracks.bpm"
    db_fields: list[str] | None = None  # multiple fields for llm criteria

    # Threshold values (deterministic rules)
    threshold: float | None = None
    threshold_min: float | None = None
    threshold_max: float | None = None
    value: str | None = None  # for exists rule

    # LLM scoring
    prompt_hint: str | None = None

    # Display
    unit: str | None = None
    fail_message: str | None = None

    @property
    def is_deterministic(self) -> bool:
        return self.rule in DETERMINISTIC_RULES

    @property
    def is_llm(self) -> bool:
        return self.rule in LLM_RULES

    @property
    def fields(self) -> list[str]:
        """Return all DB fields this criterion uses, as a list."""
        if self.db_fields:
            return self.db_fields
        if self.db_field:
            return [self.db_field]
        return []


@dataclass
class ModeConfig:
    """A loaded criteria mode configuration."""

    name: str
    description: str
    system_prompt: str
    criteria: list[Criterion]

    def get(self, criterion_id: str) -> Criterion | None:
        """Look up a criterion by id."""
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        return None


@dataclass
class EvaluationResult:
    """Result of evaluating a single criterion against a track."""

    criterion_id: str
    rule: str
    passed: bool
    score: float  # 0.0–1.0; None-equivalent is 0.0 for deterministic
    raw_value: float | None  # the actual DB value that was compared
    reasoning: str | None  # null for deterministic; set by llm.py for llm rules
    needs_llm: bool  # True if this criterion requires an LLM call


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mode(
    mode: str,
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
) -> ModeConfig:
    """
    Load and validate a criteria mode from its TOML file.

    Args:
        mode: Mode name, e.g. "1988", "contemporary", "1920s_1930s".
        config_dir: Directory containing criteria_*.toml files.

    Returns:
        ModeConfig with all criteria loaded and validated.

    Raises:
        FileNotFoundError: if the TOML file does not exist.
        ValueError: if the TOML file fails validation.
    """
    config_dir = Path(config_dir)
    toml_path = config_dir / f"criteria_{mode}.toml"

    if not toml_path.exists():
        raise FileNotFoundError(
            f"Criteria file not found: {toml_path}\nAvailable modes: {_list_available_modes(config_dir)}"
        )

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    return _parse_and_validate(raw, toml_path)


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
        score is 1.0 for pass, 0.0 for fail (or proportional for range).
        needs_llm is always False.
    """
    if criterion.is_llm:
        raise ValueError(f"evaluate_deterministic called on llm criterion '{criterion.id}'")

    raw_value = _fetch_field_value(criterion, track_id, db_path)

    if raw_value is None:
        # Null field — cannot evaluate, treat as fail with note
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

    The placeholder has needs_llm=True and score=None. The scoring
    orchestrator replaces this with the real result after the LLM call.

    Args:
        criterion: An llm-rule criterion.

    Returns:
        EvaluationResult with needs_llm=True.
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

    Returns a dict keyed by criterion_id. LLM criteria are included
    as placeholders (needs_llm=True).

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
                logger.error("Failed to evaluate criterion '%s': %s", criterion.id, e, exc_info=True)
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

    Only includes criteria that have been fully scored (needs_llm=False
    or score has been filled in by llm.py).

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
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse_and_validate(raw: dict, source_path: Path) -> ModeConfig:
    """Parse raw TOML dict into a validated ModeConfig."""
    errors: list[str] = []

    # Mode section
    mode_raw = raw.get("mode", {})
    mode_name = mode_raw.get("name", "")
    mode_desc = mode_raw.get("description", "")
    llm_context = mode_raw.get("llm_context", {})
    system_prompt = llm_context.get("system", "")

    if not mode_name:
        errors.append("mode.name is required")

    if not system_prompt:
        errors.append("mode.llm_context.system is required")

    # Criteria
    raw_criteria = raw.get("criterion", [])
    criteria: list[Criterion] = []
    seen_ids: set[str] = set()

    for i, rc in enumerate(raw_criteria):
        criterion_errors, criterion = _parse_criterion(rc, i)
        errors.extend(criterion_errors)
        if criterion is not None:
            if criterion.id in seen_ids:
                errors.append(f"Duplicate criterion id: '{criterion.id}'")
            else:
                seen_ids.add(criterion.id)
                criteria.append(criterion)

    if errors:
        error_list = "\n  - ".join(errors)
        raise ValueError(f"Criteria validation failed for {source_path}:\n  - {error_list}")

    return ModeConfig(
        name=mode_name,
        description=mode_desc,
        system_prompt=system_prompt.strip(),
        criteria=criteria,
    )


def _parse_criterion(rc: dict, index: int) -> tuple[list[str], "Criterion | None"]:
    """Parse and validate a single criterion dict. Returns (errors, criterion)."""
    errors: list[str] = []
    cid = rc.get("id", f"<criterion[{index}]>")

    # Required fields
    for req in ("id", "name", "description", "weight", "rule"):
        if req not in rc:
            errors.append(f"criterion '{cid}': missing required field '{req}'")

    if errors:
        return errors, None

    rule = rc["rule"]
    if rule not in ALL_RULES:
        errors.append(f"criterion '{cid}': unknown rule '{rule}'. Must be one of: {sorted(ALL_RULES)}")
        return errors, None

    weight = rc.get("weight", 1.0)
    if not isinstance(weight, (int, float)) or weight <= 0:
        errors.append(f"criterion '{cid}': weight must be a positive number")

    # db_field / db_fields mutual exclusivity
    has_db_field = "db_field" in rc
    has_db_fields = "db_fields" in rc

    if has_db_field and has_db_fields:
        errors.append(f"criterion '{cid}': db_field and db_fields are mutually exclusive")
        return errors, None

    if not has_db_field and not has_db_fields:
        errors.append(f"criterion '{cid}': must have either db_field or db_fields")
        return errors, None

    # Rule-specific validation
    if rule in ("lte", "gte", "eq"):
        if not has_db_field:
            errors.append(f"criterion '{cid}': rule '{rule}' requires db_field (not db_fields)")
        if "threshold" not in rc:
            errors.append(f"criterion '{cid}': rule '{rule}' requires threshold")

    elif rule == "range":
        if not has_db_field:
            errors.append(f"criterion '{cid}': rule 'range' requires db_field")
        if "threshold_min" not in rc or "threshold_max" not in rc:
            errors.append(f"criterion '{cid}': rule 'range' requires threshold_min and threshold_max")

    elif rule == "exists":
        if not has_db_field:
            errors.append(f"criterion '{cid}': rule 'exists' requires db_field")
        if "value" not in rc:
            errors.append(f"criterion '{cid}': rule 'exists' requires value")

    elif rule == "llm":
        if "prompt_hint" not in rc:
            errors.append(f"criterion '{cid}': rule 'llm' requires prompt_hint")

    if errors:
        return errors, None

    return [], Criterion(
        id=rc["id"],
        name=rc["name"],
        description=rc.get("description", ""),
        weight=float(rc["weight"]),
        rule=rule,
        db_field=rc.get("db_field"),
        db_fields=rc.get("db_fields"),
        threshold=rc.get("threshold"),
        threshold_min=rc.get("threshold_min"),
        threshold_max=rc.get("threshold_max"),
        value=rc.get("value"),
        prompt_hint=rc.get("prompt_hint"),
        unit=rc.get("unit"),
        fail_message=rc.get("fail_message"),
    )


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def _apply_rule(
    criterion: Criterion,
    raw_value: float,
) -> tuple[bool, float]:
    """
    Apply a deterministic rule and return (passed, score).

    Score is 1.0 for pass, 0.0 for fail, except range which gives
    a proportional score based on distance from bounds.

    Args:
        criterion: The criterion with rule and thresholds.
        raw_value: The actual field value from the database.

    Returns:
        (passed, score) where score is 0.0–1.0.
    """
    rule = criterion.rule

    if rule == "lte":
        passed = raw_value <= criterion.threshold
        return passed, 1.0 if passed else 0.0

    elif rule == "gte":
        passed = raw_value >= criterion.threshold
        return passed, 1.0 if passed else 0.0

    elif rule == "eq":
        passed = abs(raw_value - criterion.threshold) < 1e-9
        return passed, 1.0 if passed else 0.0

    elif rule == "range":
        lo, hi = criterion.threshold_min, criterion.threshold_max
        passed = lo <= raw_value <= hi
        if passed:
            return True, 1.0
        # Proportional score: distance from nearest bound
        if raw_value < lo:
            dist = (lo - raw_value) / (lo + 1e-9)
        else:
            dist = (raw_value - hi) / (hi + 1e-9)
        score = float(max(0.0, 1.0 - dist))
        return False, score

    elif rule == "exists":
        # raw_value for exists is 1.0 (found) or 0.0 (not found)
        passed = raw_value >= 1.0
        return passed, 1.0 if passed else 0.0

    return False, 0.0


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

    Handles all deterministic rule types:
    - tracks.* — simple column lookup on tracks table
    - sections.* — for exists rules, returns 1.0 if matching row exists
    - sections.duration WHERE label=... — for intro_length criteria

    Args:
        criterion: Criterion with db_field and rule.
        track_id: Track to query.
        db_path: SQLite database path.

    Returns:
        Float value, or None if the field is null or not found.
    """
    db_field = criterion.db_field

    conn = get_connection(db_path)
    try:
        if criterion.rule == "exists":
            # Check whether a sections row with this label exists
            count = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE track_id = ? AND label = ?",
                (track_id, criterion.value),
            ).fetchone()[0]
            return float(count)

        # Parse table.column
        if "." not in db_field:
            logger.warning("db_field '%s' has no table prefix", db_field)
            return None

        table, column = db_field.split(".", 1)

        # Handle sections.duration with WHERE clause pattern
        # e.g. "sections.duration WHERE label='intro'"
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
                # e.g. get duration of the intro section
                query = f"SELECT {column} FROM sections WHERE track_id = ? AND {where_clause} ORDER BY position LIMIT 1"
                row = conn.execute(query, (track_id,)).fetchone()
            else:
                # First section value
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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _list_available_modes(config_dir: Path) -> list[str]:
    """Return list of available mode names from config directory."""
    if not config_dir.exists():
        return []
    return [p.stem.replace("criteria_", "") for p in config_dir.glob("criteria_*.toml")]
