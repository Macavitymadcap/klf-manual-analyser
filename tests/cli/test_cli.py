"""Tests for cli/__init__.py"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from manual_analyser.cli import app
from manual_analyser.pipeline.types import RunSummary, TrackState, TrackStatus

runner = CliRunner()

_PATCH_PIPELINE = "manual_analyser.cli.run_pipeline"
_PATCH_FFMPEG = "manual_analyser.cli.check_ffmpeg"
_PATCH_OLLAMA = "manual_analyser.cli.check_ollama"
_PATCH_RENDER = "manual_analyser.cli.render"
_PATCH_SERVER = "manual_analyser.cli.run_server"


def _complete_summary(mp3_path: str) -> RunSummary:
    state = TrackState(track_id="a" * 32, mp3_path=mp3_path, status=TrackStatus.COMPLETE)
    summary = RunSummary()
    summary.complete.append(state)
    return summary


@pytest.fixture
def mp3_dir(tmp_path):
    mp3 = tmp_path / "The_KLF-Doctorin.mp3"
    mp3.write_bytes(b"\xff\xfb")
    return tmp_path


class TestAnalyseCommand:
    def test_runs_successfully(self, mp3_dir):
        with (
            patch(_PATCH_FFMPEG),
            patch(_PATCH_OLLAMA),
            patch(_PATCH_PIPELINE, return_value=_complete_summary(str(mp3_dir / "The_KLF-Doctorin.mp3"))),
        ):
            result = runner.invoke(app, ["analyse", str(mp3_dir), "--mode", "1988"])
        assert result.exit_code == 0

    def test_invalid_mode_exits_1(self, mp3_dir):
        result = runner.invoke(app, ["analyse", str(mp3_dir), "--mode", "invalid"])
        assert result.exit_code == 1
        assert "Unknown mode" in result.output

    def test_missing_path_exits_1(self, tmp_path):
        missing = tmp_path / "nonexistent"
        result = runner.invoke(app, ["analyse", str(missing), "--mode", "1988"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_empty_directory_exits_1(self, tmp_path):
        with patch(_PATCH_FFMPEG), patch(_PATCH_OLLAMA):
            result = runner.invoke(app, ["analyse", str(tmp_path), "--mode", "1988"])
        assert result.exit_code == 1
        assert "No MP3" in result.output

    def test_ffmpeg_missing_exits_1(self, mp3_dir):
        from manual_analyser.audio.decode import DecodeAbortError

        with patch(_PATCH_FFMPEG, side_effect=DecodeAbortError("no ffmpeg")), patch(_PATCH_OLLAMA):
            result = runner.invoke(app, ["analyse", str(mp3_dir), "--mode", "1988"])
        assert result.exit_code == 1
        assert "no ffmpeg" in result.output

    def test_ollama_missing_exits_1(self, mp3_dir):
        from manual_analyser.scoring.llm import OllamaUnavailableError

        with patch(_PATCH_FFMPEG), patch(_PATCH_OLLAMA, side_effect=OllamaUnavailableError("no ollama")):
            result = runner.invoke(app, ["analyse", str(mp3_dir), "--mode", "1988"])
        assert result.exit_code == 1
        assert "no ollama" in result.output

    def test_summary_shown_on_completion(self, mp3_dir):
        with (
            patch(_PATCH_FFMPEG),
            patch(_PATCH_OLLAMA),
            patch(_PATCH_PIPELINE, return_value=_complete_summary(str(mp3_dir / "The_KLF-Doctorin.mp3"))),
        ):
            result = runner.invoke(app, ["analyse", str(mp3_dir), "--mode", "1988"])
        assert "Complete" in result.output
        assert result.exit_code == 0


class TestReportCommand:
    def test_exits_1_when_no_database(self, tmp_path):
        result = runner.invoke(app, ["report", "--mode", "1988", "--data-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "No database" in result.output

    def test_renders_when_db_exists(self, tmp_path):

        db = tmp_path / "manual_analyser.db"
        db.write_bytes(b"")
        with patch("manual_analyser.cli.render", return_value=tmp_path / "index.html") as mock_render:
            result = runner.invoke(app, ["report", "--mode", "1988", "--data-dir", str(tmp_path)])
        mock_render.assert_called_once()
        assert result.exit_code == 0

    def test_render_error_exits_1(self, tmp_path):
        from manual_analyser.report.render import RenderError

        db = tmp_path / "manual_analyser.db"
        db.write_bytes(b"")
        with patch("manual_analyser.cli.render", side_effect=RenderError("template broken")):
            result = runner.invoke(app, ["report", "--mode", "1988", "--data-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "template broken" in result.output

    def test_invalid_mode_exits_1(self):
        result = runner.invoke(app, ["report", "--mode", "badmode"])
        assert result.exit_code == 1


class TestServeCommand:
    def test_calls_server(self, tmp_path):
        with patch("manual_analyser.cli.run_server") as mock_serve:
            runner.invoke(app, ["serve", "--data-dir", str(tmp_path)])
        mock_serve.assert_called_once_with(data_dir=tmp_path, port=8000)

    def test_custom_port(self, tmp_path):
        with patch("manual_analyser.cli.run_server") as mock_serve:
            runner.invoke(app, ["serve", "--port", "9000", "--data-dir", str(tmp_path)])
        mock_serve.assert_called_once_with(data_dir=tmp_path, port=9000)


class TestCleanCommand:
    def test_cleans_with_yes_flag(self, tmp_path):
        with patch("manual_analyser.cli.run_clean") as mock_clean:
            result = runner.invoke(app, ["clean", "--yes", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        mock_clean.assert_called_once()

    def test_stems_flag_passed_through(self, tmp_path):
        with patch("manual_analyser.cli.run_clean") as mock_clean:
            runner.invoke(app, ["clean", "--stems", "--yes", "--data-dir", str(tmp_path)])
        _, kwargs = mock_clean.call_args
        assert kwargs["stems"] is True
        assert kwargs["features"] is False
        assert kwargs["reports"] is False

    def test_prompts_without_yes_flag(self, tmp_path):
        with patch("manual_analyser.cli.run_clean"):
            result = runner.invoke(app, ["clean", "--data-dir", str(tmp_path)], input="n\n")
        assert result.exit_code != 0
