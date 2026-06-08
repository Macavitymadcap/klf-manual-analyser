"""Tests for manual_analyser.scoring.criteria"""

from pathlib import Path

import pytest

from manual_analyser.db import get_connection
from manual_analyser.scoring.criteria import (
    Criterion,
    EvaluationResult,
    ModeConfig,
    _apply_rule,
    _parse_criterion,
    compute_overall_score,
    evaluate_deterministic,
    load_mode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "i" * 32


@pytest.fixture
def db_with_track(tmp_path):
    """DB with a tracks row and one section."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO tracks
                (track_id, filename, duration, bpm, analysis_timestamp, analysis_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (TRACK_ID, "test.mp3", 180.0, 120.0, "2025-01-01T00:00:00+00:00", "0.1.0"),
        )
        conn.execute(
            """
            INSERT INTO sections
                (track_id, position, start, end, duration, label, label_confidence, label_source)
            VALUES (?, 0, 0.0, 20.0, 20.0, 'chorus', 0.9, 'hybrid')
            """,
            (TRACK_ID,),
        )
    conn.close()
    return db_path


@pytest.fixture
def minimal_toml_path(tmp_path):
    """Write a minimal valid TOML file and return its path."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    toml_content = """
[mode]
name = "test"
description = "Test mode"

[mode.llm_context]
system = "You are a test scorer. Respond with JSON only: {\\"score\\": 5, \\"reasoning\\": \\"test\\"}"

[[criterion]]
id = "bpm"
name = "BPM Test"
description = "Test BPM criterion"
weight = 1.0
db_field = "tracks.bpm"
rule = "lte"
threshold = 135
fail_message = "BPM too high"

[[criterion]]
id = "groove"
name = "Groove Test"
description = "Test groove criterion"
weight = 2.0
db_fields = ["tracks.danceability", "tracks.beat_regularity"]
rule = "llm"
prompt_hint = "Evaluate the groove."

[[criterion]]
id = "breakdown"
name = "Breakdown Test"
description = "Breakdown must exist"
weight = 1.0
db_field = "sections.label"
rule = "exists"
value = "breakdown"
fail_message = "No breakdown"
"""
    (config_dir / "criteria_test.toml").write_text(toml_content)
    return config_dir


# ---------------------------------------------------------------------------
# Criterion dataclass
# ---------------------------------------------------------------------------


class TestCriterion:
    def test_is_deterministic(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="lte",
            db_field="tracks.bpm",
            threshold=135,
        )
        assert c.is_deterministic is True
        assert c.is_llm is False

    def test_is_llm(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="llm",
            db_fields=["tracks.a", "tracks.b"],
            prompt_hint="evaluate",
        )
        assert c.is_llm is True
        assert c.is_deterministic is False

    def test_fields_from_db_field(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="lte",
            db_field="tracks.bpm",
            threshold=135,
        )
        assert c.fields == ["tracks.bpm"]

    def test_fields_from_db_fields(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="llm",
            db_fields=["tracks.a", "tracks.b"],
            prompt_hint="hint",
        )
        assert c.fields == ["tracks.a", "tracks.b"]


# ---------------------------------------------------------------------------
# _apply_rule
# ---------------------------------------------------------------------------


class TestApplyRule:
    def _make(self, rule, **kwargs):
        return Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule=rule,
            db_field="tracks.bpm",
            **kwargs,
        )

    def test_lte_pass(self):
        c = self._make("lte", threshold=135.0)
        passed, score = _apply_rule(c, 120.0)
        assert passed is True
        assert score == pytest.approx(1.0)

    def test_lte_fail(self):
        c = self._make("lte", threshold=135.0)
        passed, score = _apply_rule(c, 140.0)
        assert passed is False
        assert score == pytest.approx(0.0)

    def test_lte_exact_boundary_passes(self):
        c = self._make("lte", threshold=135.0)
        passed, score = _apply_rule(c, 135.0)
        assert passed is True

    def test_gte_pass(self):
        c = self._make("gte", threshold=0.15)
        passed, score = _apply_rule(c, 0.20)
        assert passed is True

    def test_gte_fail(self):
        c = self._make("gte", threshold=0.15)
        passed, score = _apply_rule(c, 0.10)
        assert passed is False

    def test_eq_pass(self):
        c = self._make("eq", threshold=4.0)
        passed, score = _apply_rule(c, 4.0)
        assert passed is True

    def test_eq_fail(self):
        c = self._make("eq", threshold=4.0)
        passed, score = _apply_rule(c, 3.0)
        assert passed is False

    def test_range_pass(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="range",
            db_field="tracks.bpm",
            threshold_min=8.0,
            threshold_max=30.0,
        )
        passed, score = _apply_rule(c, 20.0)
        assert passed is True
        assert score == pytest.approx(1.0)

    def test_range_fail_below(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="range",
            db_field="tracks.bpm",
            threshold_min=8.0,
            threshold_max=30.0,
        )
        passed, score = _apply_rule(c, 4.0)
        assert passed is False
        assert 0.0 <= score < 1.0

    def test_range_fail_above(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="range",
            db_field="tracks.bpm",
            threshold_min=8.0,
            threshold_max=30.0,
        )
        passed, score = _apply_rule(c, 60.0)
        assert passed is False

    def test_exists_pass(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="exists",
            db_field="sections.label",
            value="chorus",
        )
        passed, score = _apply_rule(c, 1.0)
        assert passed is True

    def test_exists_fail(self):
        c = Criterion(
            id="x",
            name="x",
            description="x",
            weight=1.0,
            rule="exists",
            db_field="sections.label",
            value="breakdown",
        )
        passed, score = _apply_rule(c, 0.0)
        assert passed is False


# ---------------------------------------------------------------------------
# _parse_criterion
# ---------------------------------------------------------------------------


class TestParseCriterion:
    def test_valid_lte(self):
        rc = {
            "id": "bpm",
            "name": "BPM",
            "description": "test",
            "weight": 1.5,
            "rule": "lte",
            "db_field": "tracks.bpm",
            "threshold": 135,
        }
        errors, c = _parse_criterion(rc, 0)
        assert errors == []
        assert c is not None
        assert c.id == "bpm"
        assert c.threshold == 135

    def test_valid_llm_with_db_fields(self):
        rc = {
            "id": "groove",
            "name": "Groove",
            "description": "test",
            "weight": 2.0,
            "rule": "llm",
            "db_fields": ["tracks.danceability", "tracks.beat_regularity"],
            "prompt_hint": "Evaluate groove.",
        }
        errors, c = _parse_criterion(rc, 0)
        assert errors == []
        assert c.db_fields == ["tracks.danceability", "tracks.beat_regularity"]

    def test_missing_id_returns_error(self):
        rc = {
            "name": "x",
            "description": "x",
            "weight": 1.0,
            "rule": "lte",
            "db_field": "tracks.bpm",
            "threshold": 135,
        }
        errors, c = _parse_criterion(rc, 0)
        assert any("id" in e for e in errors)

    def test_both_db_field_and_db_fields_returns_error(self):
        rc = {
            "id": "x",
            "name": "x",
            "description": "x",
            "weight": 1.0,
            "rule": "llm",
            "db_field": "tracks.bpm",
            "db_fields": ["tracks.bpm"],
            "prompt_hint": "hint",
        }
        errors, c = _parse_criterion(rc, 0)
        assert any("mutually exclusive" in e for e in errors)

    def test_lte_without_threshold_returns_error(self):
        rc = {
            "id": "x",
            "name": "x",
            "description": "x",
            "weight": 1.0,
            "rule": "lte",
            "db_field": "tracks.bpm",
        }
        errors, c = _parse_criterion(rc, 0)
        assert any("threshold" in e for e in errors)

    def test_exists_without_value_returns_error(self):
        rc = {
            "id": "x",
            "name": "x",
            "description": "x",
            "weight": 1.0,
            "rule": "exists",
            "db_field": "sections.label",
        }
        errors, c = _parse_criterion(rc, 0)
        assert any("value" in e for e in errors)

    def test_unknown_rule_returns_error(self):
        rc = {
            "id": "x",
            "name": "x",
            "description": "x",
            "weight": 1.0,
            "rule": "frobulate",
            "db_field": "tracks.bpm",
        }
        errors, c = _parse_criterion(rc, 0)
        assert any("unknown rule" in e for e in errors)


# ---------------------------------------------------------------------------
# load_mode
# ---------------------------------------------------------------------------


class TestLoadMode:
    def test_loads_valid_toml(self, minimal_toml_path):
        config = load_mode("test", config_dir=minimal_toml_path)
        assert config.name == "test"
        assert len(config.criteria) == 3

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mode("nonexistent", config_dir=tmp_path)

    def test_criteria_have_correct_rules(self, minimal_toml_path):
        config = load_mode("test", config_dir=minimal_toml_path)
        bpm = config.get("bpm")
        assert bpm is not None
        assert bpm.rule == "lte"
        assert bpm.threshold == 135

    def test_llm_criterion_loaded(self, minimal_toml_path):
        config = load_mode("test", config_dir=minimal_toml_path)
        groove = config.get("groove")
        assert groove is not None
        assert groove.rule == "llm"
        assert groove.db_fields == ["tracks.danceability", "tracks.beat_regularity"]

    def test_system_prompt_loaded(self, minimal_toml_path):
        config = load_mode("test", config_dir=minimal_toml_path)
        assert "scorer" in config.system_prompt

    def test_raises_on_validation_error(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        bad_toml = """
[mode]
name = "bad"
[mode.llm_context]
system = "test"
[[criterion]]
id = "bad"
name = "bad"
description = "bad"
weight = 1.0
rule = "lte"
db_field = "tracks.bpm"
# missing threshold
"""
        (config_dir / "criteria_bad.toml").write_text(bad_toml)
        with pytest.raises(ValueError) as exc_info:
            load_mode("bad", config_dir=config_dir)
        assert "threshold" in str(exc_info.value)


# ---------------------------------------------------------------------------
# evaluate_deterministic
# ---------------------------------------------------------------------------


class TestEvaluateDeterministic:
    def _bpm_criterion(self):
        return Criterion(
            id="bpm",
            name="BPM",
            description="test",
            weight=1.5,
            rule="lte",
            db_field="tracks.bpm",
            threshold=135,
            fail_message="Too fast",
        )

    def test_pass_returns_score_1(self, db_with_track, tmp_path):
        c = self._bpm_criterion()
        result = evaluate_deterministic(c, TRACK_ID, db_with_track)
        assert result.passed is True
        assert result.score == pytest.approx(1.0)
        assert result.raw_value == pytest.approx(120.0)
        assert result.needs_llm is False

    def test_fail_returns_score_0(self, db_with_track, tmp_path):
        c = Criterion(
            id="bpm",
            name="BPM",
            description="test",
            weight=1.5,
            rule="lte",
            db_field="tracks.bpm",
            threshold=100,
            fail_message="Too fast",
        )
        result = evaluate_deterministic(c, TRACK_ID, db_with_track)
        assert result.passed is False
        assert result.score == pytest.approx(0.0)
        assert result.reasoning == "Too fast"

    def test_null_field_returns_fail(self, db_with_track):
        # danceability is null in this DB
        c = Criterion(
            id="dance",
            name="Dance",
            description="test",
            weight=1.0,
            rule="gte",
            db_field="tracks.danceability",
            threshold=0.5,
        )
        result = evaluate_deterministic(c, TRACK_ID, db_with_track)
        assert result.passed is False
        assert result.raw_value is None
        assert "null" in result.reasoning.lower()

    def test_exists_pass(self, db_with_track):
        # sections has a chorus row
        c = Criterion(
            id="chorus",
            name="Chorus",
            description="test",
            weight=1.0,
            rule="exists",
            db_field="sections.label",
            value="chorus",
        )
        result = evaluate_deterministic(c, TRACK_ID, db_with_track)
        assert result.passed is True

    def test_exists_fail(self, db_with_track):
        # no breakdown row
        c = Criterion(
            id="breakdown",
            name="Breakdown",
            description="test",
            weight=1.0,
            rule="exists",
            db_field="sections.label",
            value="breakdown",
        )
        result = evaluate_deterministic(c, TRACK_ID, db_with_track)
        assert result.passed is False

    def test_raises_for_llm_criterion(self, db_with_track):
        c = Criterion(
            id="groove",
            name="Groove",
            description="test",
            weight=2.0,
            rule="llm",
            db_fields=["tracks.danceability"],
            prompt_hint="evaluate",
        )
        with pytest.raises(ValueError, match="llm criterion"):
            evaluate_deterministic(c, TRACK_ID, db_with_track)


# ---------------------------------------------------------------------------
# compute_overall_score
# ---------------------------------------------------------------------------


class TestComputeOverallScore:
    def _mode(self, criteria):
        return ModeConfig(name="test", description="test", system_prompt="test", criteria=criteria)

    def test_all_pass_returns_1(self):
        criteria = [
            Criterion(
                id="a",
                name="a",
                description="a",
                weight=1.0,
                rule="lte",
                db_field="tracks.bpm",
                threshold=135,
            ),
            Criterion(
                id="b",
                name="b",
                description="b",
                weight=1.0,
                rule="lte",
                db_field="tracks.bpm",
                threshold=135,
            ),
        ]
        results = {
            "a": EvaluationResult("a", "lte", True, 1.0, 120.0, None, False),
            "b": EvaluationResult("b", "lte", True, 1.0, 120.0, None, False),
        }
        score = compute_overall_score(results, self._mode(criteria))
        assert score == pytest.approx(1.0)

    def test_all_fail_returns_0(self):
        criteria = [
            Criterion(
                id="a",
                name="a",
                description="a",
                weight=1.0,
                rule="lte",
                db_field="tracks.bpm",
                threshold=135,
            ),
        ]
        results = {
            "a": EvaluationResult("a", "lte", False, 0.0, 140.0, "fail", False),
        }
        score = compute_overall_score(results, self._mode(criteria))
        assert score == pytest.approx(0.0)

    def test_weighted_average(self):
        criteria = [
            Criterion(
                id="a",
                name="a",
                description="a",
                weight=2.0,
                rule="lte",
                db_field="tracks.bpm",
                threshold=135,
            ),
            Criterion(
                id="b",
                name="b",
                description="b",
                weight=1.0,
                rule="lte",
                db_field="tracks.bpm",
                threshold=135,
            ),
        ]
        results = {
            "a": EvaluationResult("a", "lte", True, 1.0, 120.0, None, False),
            "b": EvaluationResult("b", "lte", False, 0.0, 140.0, "fail", False),
        }
        # (1.0 * 2.0 + 0.0 * 1.0) / 3.0 = 0.667
        score = compute_overall_score(results, self._mode(criteria))
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_llm_placeholders_excluded(self):
        """needs_llm=True criteria should not affect the score."""
        criteria = [
            Criterion(
                id="a",
                name="a",
                description="a",
                weight=1.0,
                rule="lte",
                db_field="tracks.bpm",
                threshold=135,
            ),
            Criterion(
                id="b",
                name="b",
                description="b",
                weight=2.0,
                rule="llm",
                db_fields=["tracks.danceability"],
                prompt_hint="test",
            ),
        ]
        results = {
            "a": EvaluationResult("a", "lte", True, 1.0, 120.0, None, False),
            "b": EvaluationResult("b", "llm", False, 0.0, None, None, True),
        }
        score = compute_overall_score(results, self._mode(criteria))
        assert score == pytest.approx(1.0)  # only criterion a counted

    def test_empty_results_returns_zero(self):
        criteria = [
            Criterion(
                id="a",
                name="a",
                description="a",
                weight=1.0,
                rule="lte",
                db_field="tracks.bpm",
                threshold=135,
            ),
        ]
        score = compute_overall_score({}, self._mode(criteria))
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: load real TOML files
# ---------------------------------------------------------------------------


class TestLoadRealTomls:
    def test_loads_1988_mode(self):
        """Verify the real criteria_1988.toml loads without errors."""
        config_dir = Path("config")
        if not config_dir.exists():
            pytest.skip("config/ directory not found — run from project root")
        config = load_mode("1988", config_dir=config_dir)
        assert config.name == "1988"
        assert len(config.criteria) > 0
        # Verify key criteria exist
        assert config.get("bpm") is not None
        assert config.get("groove") is not None
        assert config.get("structure") is not None

    def test_loads_contemporary_mode(self):
        config_dir = Path("config")
        if not config_dir.exists():
            pytest.skip("config/ directory not found — run from project root")
        config = load_mode("contemporary", config_dir=config_dir)
        assert config.name == "contemporary"
        assert config.get("hook_timing") is not None

    def test_loads_1920s_mode(self):
        config_dir = Path("config")
        if not config_dir.exists():
            pytest.skip("config/ directory not found — run from project root")
        config = load_mode("1920s_1930s", config_dir=config_dir)
        assert config.name == "1920s_1930s"
        assert config.get("aaba_form") is not None
