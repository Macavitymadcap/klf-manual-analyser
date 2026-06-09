"""
scoring/loader.py — Load and validate criteria TOML config files.

Public API:
    load_mode(mode, config_dir) -> ModeConfig

The TOML structure expected:

    [mode]
    name = "1988"
    description = "..."
    [mode.llm_context]
    system = "You are a scoring assistant..."

    [[criterion]]
    id = "bpm"
    name = "BPM"
    description = "..."
    weight = 1.5
    rule = "lte"
    db_field = "tracks.bpm"
    threshold = 135

    [[criterion]]
    id = "groove"
    ...
    rule = "llm"
    db_fields = ["tracks.danceability", "tracks.beat_regularity"]
    prompt_hint = "Evaluate the groove..."
"""

import logging
import tomllib
from pathlib import Path

from manual_analyser.scoring.types import (
    ALL_RULES,
    Criterion,
    ModeConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path("config")


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


# ---------------------------------------------------------------------------
# Internal — parsing and validation
# ---------------------------------------------------------------------------


def _parse_and_validate(raw: dict, source_path: Path) -> ModeConfig:
    """Parse raw TOML dict into a validated ModeConfig."""
    errors: list[str] = []

    mode_raw = raw.get("mode", {})
    mode_name = mode_raw.get("name", "")
    mode_desc = mode_raw.get("description", "")
    llm_context = mode_raw.get("llm_context", {})
    system_prompt = llm_context.get("system", "")

    if not mode_name:
        errors.append("mode.name is required")
    if not system_prompt:
        errors.append("mode.llm_context.system is required")

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

    has_db_field = "db_field" in rc
    has_db_fields = "db_fields" in rc

    if has_db_field and has_db_fields:
        errors.append(f"criterion '{cid}': db_field and db_fields are mutually exclusive")
        return errors, None

    if not has_db_field and not has_db_fields:
        errors.append(f"criterion '{cid}': must have either db_field or db_fields")
        return errors, None

    errors = _validate_rules(rule, has_db_field, rc, cid, errors)

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


def _validate_rules(
    rule: str,
    has_db_field: bool,
    rc: dict,
    cid: str,
    errors: list[str],
) -> list[str]:
    errors = _validate_number(rule, has_db_field, cid, rc, errors)
    errors = _validate_range(rule, has_db_field, cid, rc, errors)
    errors = _validate_exists(rule, has_db_field, cid, rc, errors)

    if rule == "llm" and "prompt_hint" not in rc:
        errors.append(f"criterion '{cid}': rule 'llm' requires prompt_hint")

    return errors


def _validate_number(rule: str, has_db_field: bool, cid: str, rc: dict, errors: list[str]) -> list[str]:
    if rule in ("lte", "gte", "eq"):
        if not has_db_field:
            errors.append(f"criterion '{cid}': rule '{rule}' requires db_field (not db_fields)")
        if "threshold" not in rc:
            errors.append(f"criterion '{cid}': rule '{rule}' requires threshold")
    return errors


def _validate_range(rule: str, has_db_field: bool, cid: str, rc: dict, errors: list[str]) -> list[str]:
    if rule == "range":
        if not has_db_field:
            errors.append(f"criterion '{cid}': rule 'range' requires db_field")
        if "threshold_min" not in rc or "threshold_max" not in rc:
            errors.append(f"criterion '{cid}': rule 'range' requires threshold_min and threshold_max")
    return errors


def _validate_exists(rule: str, has_db_field: bool, cid: str, rc: dict, errors: list[str]) -> list[str]:
    if rule == "exists":
        if not has_db_field:
            errors.append(f"criterion '{cid}': rule 'exists' requires db_field")
        if "value" not in rc:
            errors.append(f"criterion '{cid}': rule 'exists' requires value")
    return errors


def _list_available_modes(config_dir: Path) -> list[str]:
    """Return list of available mode names from config directory."""
    if not config_dir.exists():
        return []
    return [p.stem.replace("criteria_", "") for p in config_dir.glob("criteria_*.toml")]
