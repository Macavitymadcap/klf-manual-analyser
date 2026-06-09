"""
scoring/types.py — Core dataclasses for the scoring layer.

Criterion     — a single scoring rule loaded from a TOML config file
ModeConfig    — a full loaded mode (system prompt + list of Criterion)
EvaluationResult — the result of evaluating one criterion against one track
"""

from dataclasses import dataclass

# Valid rule type sets — imported here so loader and evaluator can both use them
DETERMINISTIC_RULES: frozenset[str] = frozenset({"lte", "gte", "eq", "range", "exists"})
LLM_RULES: frozenset[str] = frozenset({"llm"})
ALL_RULES: frozenset[str] = DETERMINISTIC_RULES | LLM_RULES


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
    score: float  # 0.0–1.0; placeholder 0.0 for needs_llm=True
    raw_value: float | None  # the actual DB value that was compared
    reasoning: str | None  # null for deterministic; set by llm.py for llm rules
    needs_llm: bool  # True if this criterion still requires an LLM call


@dataclass
class PromptPackage:
    """A complete prompt ready to send to the LLM."""

    criterion_id: str
    system_prompt: str
    user_prompt: str
    has_null_fields: bool  # True if any field values are null
    null_field_names: list[str]  # Names of null fields for logging
