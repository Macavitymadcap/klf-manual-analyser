"""Tests for pipeline/__init__.py"""

from unittest.mock import patch

import pytest

from manual_analyser.audio.decode import DecodeAbortError, DecodeResult, DecodeSkipError
from manual_analyser.audio.separate import SeparateAbortError, SeparateResult, SeparateSkipError
from manual_analyser.pipeline import RunSummary, run_pipeline

_TRACK_ID = "a" * 32
_DECODE = "manual_analyser.pipeline.decode_track"
_SEPARATE = "manual_analyser.pipeline.separate_track"
_ANALYSE = "manual_analyser.pipeline.analysis_runner.run_analysis"
_TRANSCRIBE = "manual_analyser.pipeline.transcribe_track"
_ALIGN = "manual_analyser.pipeline.align_sections"
_EMBED = "manual_analyser.pipeline.embed_track"
_SCORE = "manual_analyser.pipeline.scoring_runner.run_scoring"
_QDRANT = "manual_analyser.pipeline.is_qdrant_available"


def _fake_decode_result(tmp_path) -> DecodeResult:
    full_wav = tmp_path / "full.wav"
    full_wav.write_bytes(b"RIFF")
    return DecodeResult(
        track_id=_TRACK_ID,
        filename="test.mp3",
        artist="Test",
        song_name="Track",
        duration=180.0,
        stem_dir=tmp_path,
        full_wav=full_wav,
    )


def _fake_separate_result(tmp_path) -> SeparateResult:
    for name in ("drums", "bass", "vocals", "other"):
        (tmp_path / f"{name}.wav").write_bytes(b"RIFF")
    return SeparateResult(
        track_id=_TRACK_ID,
        stem_dir=tmp_path,
        drums=tmp_path / "drums.wav",
        bass=tmp_path / "bass.wav",
        vocals=tmp_path / "vocals.wav",
        other=tmp_path / "other.wav",
        device_used="cpu",
        was_cpu_fallback=False,
    )


@pytest.fixture
def db_path(tmp_path):
    from manual_analyser.db import get_connection

    path = tmp_path / "test.db"
    get_connection(path).close()
    return path


@pytest.fixture
def mp3(tmp_path):
    p = tmp_path / "Test-Track.mp3"
    p.write_bytes(b"\xff\xfb")
    return p


class TestRunPipelineSuccess:
    def test_returns_run_summary(self, mp3, db_path, tmp_path):
        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)),
            patch(_SEPARATE, return_value=_fake_separate_result(tmp_path)),
            patch(_ANALYSE),
            patch(_TRANSCRIBE),
            patch(_ALIGN),
            patch(_EMBED),
            patch(_SCORE),
            patch(_QDRANT, return_value=False),
        ):
            result = run_pipeline([mp3], "1988", db_path, tmp_path)
        assert isinstance(result, RunSummary)

    def test_complete_track_in_summary(self, mp3, db_path, tmp_path):
        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)),
            patch(_SEPARATE, return_value=_fake_separate_result(tmp_path)),
            patch(_ANALYSE),
            patch(_TRANSCRIBE),
            patch(_ALIGN),
            patch(_EMBED),
            patch(_SCORE),
            patch(_QDRANT, return_value=False),
        ):
            result = run_pipeline([mp3], "1988", db_path, tmp_path)
        assert len(result.complete) == 1
        assert result.complete[0].track_id == _TRACK_ID

    def test_all_stages_called(self, mp3, db_path, tmp_path):
        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)) as d,
            patch(_SEPARATE, return_value=_fake_separate_result(tmp_path)) as s,
            patch(_ANALYSE) as a,
            patch(_TRANSCRIBE) as t,
            patch(_ALIGN) as al,
            patch(_SCORE) as sc,
            patch(_EMBED),
            patch(_QDRANT, return_value=False),
        ):
            run_pipeline([mp3], "1988", db_path, tmp_path, no_cache=True)
        d.assert_called_once()
        s.assert_called_once()
        a.assert_called_once()
        t.assert_called_once()
        al.assert_called_once()
        sc.assert_called_once()


class TestRunPipelineSkip:
    def test_decode_skip_error_marks_track_skipped(self, mp3, db_path, tmp_path):
        with patch(_DECODE, side_effect=DecodeSkipError("bad file")), patch(_QDRANT, return_value=False):
            result = run_pipeline([mp3], "1988", db_path, tmp_path)
        assert len(result.skipped) == 1
        assert "decode_failed" in result.skipped[0].notes[0]

    def test_separate_skip_error_marks_track_skipped(self, mp3, db_path, tmp_path):
        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)),
            patch(_SEPARATE, side_effect=SeparateSkipError("oom")),
            patch(_QDRANT, return_value=False),
        ):
            result = run_pipeline([mp3], "1988", db_path, tmp_path)
        assert len(result.skipped) == 1

    def test_decode_abort_error_propagates(self, mp3, db_path, tmp_path):
        with patch(_DECODE, side_effect=DecodeAbortError("no ffmpeg")), patch(_QDRANT, return_value=False):
            with pytest.raises(DecodeAbortError):
                run_pipeline([mp3], "1988", db_path, tmp_path)

    def test_separate_abort_error_propagates(self, mp3, db_path, tmp_path):
        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)),
            patch(_SEPARATE, side_effect=SeparateAbortError("no demucs")),
            patch(_QDRANT, return_value=False),
        ):
            with pytest.raises(SeparateAbortError):
                run_pipeline([mp3], "1988", db_path, tmp_path)


class TestRunPipelinePartial:
    def test_analysis_failure_marks_partial(self, mp3, db_path, tmp_path):
        def bad_analyse(stems, db_path, state):
            state.failed_stages.append("tempo")

        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)),
            patch(_SEPARATE, return_value=_fake_separate_result(tmp_path)),
            patch(_ANALYSE, side_effect=bad_analyse),
            patch(_TRANSCRIBE),
            patch(_ALIGN),
            patch(_SCORE),
            patch(_EMBED),
            patch(_QDRANT, return_value=False),
        ):
            result = run_pipeline([mp3], "1988", db_path, tmp_path, no_cache=True)
        assert len(result.partial) == 1

    def test_multiple_tracks_isolated(self, db_path, tmp_path):
        mp3_a = tmp_path / "A-Song.mp3"
        mp3_b = tmp_path / "B-Song.mp3"
        mp3_a.write_bytes(b"\xff\xfb")
        mp3_b.write_bytes(b"\xff\xfb")

        call_count = {"n": 0}

        def decode_side_effect(path, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise DecodeSkipError("corrupt")
            r = _fake_decode_result(tmp_path)
            r = DecodeResult(_TRACK_ID, path.name, None, None, 180.0, tmp_path, tmp_path / "full.wav")
            return r

        with (
            patch(_DECODE, side_effect=decode_side_effect),
            patch(_SEPARATE, return_value=_fake_separate_result(tmp_path)),
            patch(_ANALYSE),
            patch(_TRANSCRIBE),
            patch(_ALIGN),
            patch(_SCORE),
            patch(_EMBED),
            patch(_QDRANT, return_value=False),
        ):
            result = run_pipeline([mp3_a, mp3_b], "1988", db_path, tmp_path, no_cache=True)

        assert len(result.skipped) == 1
        assert len(result.complete) == 1


class TestCaching:
    def test_analysis_skipped_when_cached(self, mp3, db_path, tmp_path):
        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)),
            patch(_SEPARATE, return_value=_fake_separate_result(tmp_path)),
            patch("manual_analyser.pipeline.cache.track_in_db", return_value=True),
            patch("manual_analyser.pipeline.cache.transcript_in_db", return_value=True),
            patch("manual_analyser.pipeline.cache.sections_labelled", return_value=True),
            patch("manual_analyser.pipeline.cache.scores_exist", return_value=True),
            patch(_ANALYSE) as mock_analyse,
            patch(_TRANSCRIBE) as mock_transcribe,
            patch(_SCORE) as mock_score,
            patch(_EMBED),
            patch(_QDRANT, return_value=False),
        ):
            run_pipeline([mp3], "1988", db_path, tmp_path, no_cache=False)
        mock_analyse.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_score.assert_not_called()

    def test_no_cache_flag_bypasses_checks(self, mp3, db_path, tmp_path):
        with (
            patch(_DECODE, return_value=_fake_decode_result(tmp_path)),
            patch(_SEPARATE, return_value=_fake_separate_result(tmp_path)),
            patch("manual_analyser.pipeline.cache.track_in_db", return_value=True),
            patch(_ANALYSE) as mock_analyse,
            patch(_TRANSCRIBE),
            patch(_ALIGN),
            patch(_SCORE),
            patch(_EMBED),
            patch(_QDRANT, return_value=False),
        ):
            run_pipeline([mp3], "1988", db_path, tmp_path, no_cache=True)
        mock_analyse.assert_called_once()
