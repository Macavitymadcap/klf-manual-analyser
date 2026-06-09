"""Tests for scoring/llm/llm/response_parser.py"""

import pytest

from manual_analyser.scoring.llm.response_parser import parse, strict_prompt
from manual_analyser.scoring.llm.types import LlmScore


class TestParse:
    def test_clean_json(self):
        raw = '{"score": 7, "reasoning": "Good groove"}'
        result = parse("groove", raw)
        assert isinstance(result, LlmScore)
        assert result.raw_score == 7
        assert result.score == pytest.approx(0.7)
        assert result.reasoning == "Good groove"
        assert result.passed is True

    def test_strips_markdown_fences(self):
        raw = '```json\n{"score": 4, "reasoning": "Weak hook"}\n```'
        result = parse("chorus_hook", raw)
        assert result.raw_score == 4
        assert result.passed is False

    def test_strips_plain_fences(self):
        raw = '```\n{"score": 5, "reasoning": "OK"}\n```'
        result = parse("structure", raw)
        assert result.raw_score == 5

    def test_clamps_score_above_max(self):
        raw = '{"score": 15, "reasoning": "Too high"}'
        result = parse("bpm", raw)
        assert result.raw_score == 10

    def test_clamps_score_below_min(self):
        raw = '{"score": -3, "reasoning": "Negative"}'
        result = parse("bpm", raw)
        assert result.raw_score == 0

    def test_score_5_is_passing(self):
        raw = '{"score": 5, "reasoning": "Borderline"}'
        result = parse("groove", raw)
        assert result.passed is True

    def test_score_4_is_failing(self):
        raw = '{"score": 4, "reasoning": "Below threshold"}'
        result = parse("groove", raw)
        assert result.passed is False

    def test_missing_score_raises(self):
        raw = '{"reasoning": "forgot the score"}'
        with pytest.raises(ValueError):
            parse("groove", raw)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse("groove", "not json at all")

    def test_normalised_score(self):
        raw = '{"score": 10, "reasoning": "Perfect"}'
        result = parse("groove", raw)
        assert result.score == pytest.approx(1.0)

    def test_zero_score(self):
        raw = '{"score": 0, "reasoning": "Terrible"}'
        result = parse("groove", raw)
        assert result.score == pytest.approx(0.0)
        assert result.passed is False

    def test_criterion_id_preserved(self):
        raw = '{"score": 6, "reasoning": "Fine"}'
        result = parse("keys_harmony", raw)
        assert result.criterion_id == "keys_harmony"

    def test_missing_reasoning_defaults_empty(self):
        raw = '{"score": 7}'
        result = parse("groove", raw)
        assert result.reasoning == ""


class TestStrictPrompt:
    def test_appends_suffix(self):
        result = strict_prompt("Score this track.")
        assert "Score this track." in result
        assert '"score"' in result
        assert "integer 0-10" in result

    def test_original_prompt_preserved(self):
        original = "Here are the track features:\nbpm=120"
        result = strict_prompt(original)
        assert original in result
