"""Tests for scoring/llm/__init__.py"""

from unittest.mock import patch

import httpx
import pytest

from manual_analyser.db import get_connection
from manual_analyser.scoring.llm import LlmFailure, LlmScore, OllamaTimeoutError, score_all, score_criterion
from manual_analyser.scoring.prompt import PromptPackage

TRACK_ID = "a" * 32
MODE = "1988"

_GOOD_JSON = '{"score": 7, "reasoning": "Solid groove feel"}'
_BAD_JSON = "not json"

_PATCH = "manual_analyser.scoring.llm.ollama_client.chat"


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = get_connection(path)
    with conn:
        conn.execute(
            "INSERT INTO tracks (track_id, filename, duration, analysis_timestamp, analysis_version)"
            " VALUES (?, ?, ?, ?, ?)",
            (TRACK_ID, "test.mp3", 180.0, "2025-01-01T00:00:00+00:00", "0.1.0"),
        )
    conn.close()
    return path


@pytest.fixture
def package():
    return PromptPackage(
        criterion_id="groove",
        system_prompt="You are a music analyst.",
        user_prompt="Score the groove.",
        has_null_fields=False,
        null_field_names=[],
    )


class TestScoreCriterion:
    def test_success_returns_llm_score(self, package, db_path):
        with patch(_PATCH, return_value=_GOOD_JSON):
            result = score_criterion(package, TRACK_ID, MODE, db_path)
        assert isinstance(result, LlmScore)
        assert result.criterion_id == "groove"
        assert result.raw_score == 7

    def test_success_writes_to_db(self, package, db_path):
        with patch(_PATCH, return_value=_GOOD_JSON):
            score_criterion(package, TRACK_ID, MODE, db_path)
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT score, reasoning, passed FROM scores WHERE track_id=? AND criterion_id=?",
            (TRACK_ID, "groove"),
        ).fetchone()
        conn.close()
        assert row is not None
        assert abs(row[0] - 0.7) < 0.001
        assert row[1] == "Solid groove feel"
        assert row[2] == 1

    def test_parse_failure_retries_with_strict_prompt(self, package, db_path):
        with patch(_PATCH, side_effect=[_BAD_JSON, _GOOD_JSON]):
            result = score_criterion(package, TRACK_ID, MODE, db_path)
        assert isinstance(result, LlmScore)

    def test_parse_failure_on_both_attempts_returns_failure(self, package, db_path):
        with patch(_PATCH, return_value=_BAD_JSON):
            result = score_criterion(package, TRACK_ID, MODE, db_path)
        assert isinstance(result, LlmFailure)
        assert result.reason == "parse_error"

    def test_failure_writes_null_score_to_db(self, package, db_path):
        with patch(_PATCH, return_value=_BAD_JSON):
            score_criterion(package, TRACK_ID, MODE, db_path)
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT score, passed FROM scores WHERE track_id=? AND criterion_id=?",
            (TRACK_ID, "groove"),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] is None
        assert row[1] == 0

    def test_timeout_retries_once(self, package, db_path):
        with patch(_PATCH, side_effect=[OllamaTimeoutError("timed out"), _GOOD_JSON]):
            result = score_criterion(package, TRACK_ID, MODE, db_path)
        assert isinstance(result, LlmScore)

    def test_timeout_on_both_attempts_returns_failure(self, package, db_path):
        with patch(_PATCH, side_effect=OllamaTimeoutError("timed out")):
            result = score_criterion(package, TRACK_ID, MODE, db_path)
        assert isinstance(result, LlmFailure)
        assert result.reason == "timeout"

    def test_http_error_returns_failure(self, package, db_path):
        with patch(_PATCH, side_effect=httpx.HTTPError("500")):
            result = score_criterion(package, TRACK_ID, MODE, db_path)
        assert isinstance(result, LlmFailure)
        assert result.reason == "http_error"


class TestScoreAll:
    def test_returns_one_result_per_package(self, db_path):
        packages = [
            PromptPackage("groove", "sys", "user", False, []),
            PromptPackage("chorus_hook", "sys", "user", False, []),
        ]
        with patch(_PATCH, return_value=_GOOD_JSON):
            results = score_all(packages, TRACK_ID, MODE, db_path)
        assert len(results) == 2
        assert results[0].criterion_id == "groove"
        assert results[1].criterion_id == "chorus_hook"

    def test_failure_in_one_does_not_abort_others(self, db_path):
        packages = [
            PromptPackage("groove", "sys", "user", False, []),
            PromptPackage("chorus_hook", "sys", "user", False, []),
        ]
        with patch(_PATCH, side_effect=[_BAD_JSON, _BAD_JSON, _GOOD_JSON, _GOOD_JSON]):
            results = score_all(packages, TRACK_ID, MODE, db_path)
        assert isinstance(results[0], LlmFailure)
        assert isinstance(results[1], LlmScore)
