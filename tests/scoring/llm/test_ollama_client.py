"""Tests for scoring/llm/ollama_client.py"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from manual_analyser.scoring.llm.ollama_client import (
    OllamaTimeoutError,
    OllamaUnavailableError,
    chat,
    check_ollama,
)


class TestCheckOllama:
    def test_passes_when_model_present(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "qwen2.5:14b"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            check_ollama("qwen2.5:14b")  # should not raise

    def test_raises_when_ollama_unreachable(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OllamaUnavailableError, match="not reachable"):
                check_ollama("qwen2.5:14b")

    def test_raises_when_model_not_present(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(OllamaUnavailableError, match="not found"):
                check_ollama("qwen2.5:14b")

    def test_raises_when_model_list_empty(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(OllamaUnavailableError):
                check_ollama("qwen2.5:14b")


class TestChat:
    def _mock_response(self, content: str) -> MagicMock:
        mock = MagicMock()
        mock.json.return_value = {"message": {"content": content}}
        mock.raise_for_status = MagicMock()
        return mock

    def test_returns_content_string(self):
        with patch("httpx.post", return_value=self._mock_response("hello")):
            result = chat("sys", "user", "qwen2.5:14b")
        assert result == "hello"

    def test_raises_timeout_error(self):
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(OllamaTimeoutError):
                chat("sys", "user", "qwen2.5:14b")

    def test_raises_on_http_error(self):
        mock = MagicMock()
        mock.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        with patch("httpx.post", return_value=mock):
            with pytest.raises(httpx.HTTPStatusError):
                chat("sys", "user", "qwen2.5:14b")
