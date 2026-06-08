"""Tests for manual_analyser.audio.separate"""

import wave
from unittest.mock import patch

import numpy as np
import pytest

from manual_analyser.audio.separate import (
    STEM_NAMES,
    SeparateResult,
    SeparateSkipError,
    _build_result,
    _is_oom,
    _stems_exist,
    separate_track,
)
from manual_analyser.audio.separate import (
    _OOMError as _OOMError_for_test,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "a" * 32


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a minimal real WAV file."""
    wav_path = tmp_path / "full.wav"
    sample_rate = 44100
    samples = np.zeros(sample_rate * 2, dtype=np.int16)  # 2 seconds
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return wav_path


@pytest.fixture
def stem_dir_with_all_stems(tmp_path):
    """Create a stem directory with all four expected WAV files."""
    stem_dir = tmp_path / "stems" / TRACK_ID
    stem_dir.mkdir(parents=True)
    for name in STEM_NAMES:
        (stem_dir / f"{name}.wav").write_bytes(b"RIFF")
    return stem_dir


@pytest.fixture
def empty_stem_dir(tmp_path):
    stem_dir = tmp_path / "stems" / TRACK_ID
    stem_dir.mkdir(parents=True)
    return stem_dir


# ---------------------------------------------------------------------------
# _stems_exist
# ---------------------------------------------------------------------------


class TestStemsExist:
    def test_returns_true_when_all_present(self, stem_dir_with_all_stems):
        assert _stems_exist(stem_dir_with_all_stems) is True

    def test_returns_false_when_dir_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        assert _stems_exist(missing) is False

    def test_returns_false_when_partial(self, tmp_path):
        stem_dir = tmp_path / "stems" / TRACK_ID
        stem_dir.mkdir(parents=True)
        (stem_dir / "drums.wav").write_bytes(b"RIFF")
        # Only one stem — should be False
        assert _stems_exist(stem_dir) is False


# ---------------------------------------------------------------------------
# _is_oom
# ---------------------------------------------------------------------------


class TestIsOom:
    def test_detects_cuda_oom(self):
        e = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert _is_oom(e) is True

    def test_detects_mps_oom(self):
        e = RuntimeError("MPS backend out of memory")
        assert _is_oom(e) is True

    def test_detects_generic_oom(self):
        e = RuntimeError("out of memory")
        assert _is_oom(e) is True

    def test_ignores_unrelated_error(self):
        e = RuntimeError("Model not found")
        assert _is_oom(e) is False


# ---------------------------------------------------------------------------
# _build_result
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_builds_correct_paths(self, tmp_path):
        stem_dir = tmp_path / "stems" / TRACK_ID
        result = _build_result(TRACK_ID, stem_dir, "cuda", False)
        assert result.drums == stem_dir / "drums.wav"
        assert result.bass == stem_dir / "bass.wav"
        assert result.vocals == stem_dir / "vocals.wav"
        assert result.other == stem_dir / "other.wav"
        assert result.device_used == "cuda"
        assert result.was_cpu_fallback is False


# ---------------------------------------------------------------------------
# separate_track — cache behaviour
# ---------------------------------------------------------------------------


class TestSeparateTrackCaching:
    def test_returns_cached_result_when_stems_exist(self, tmp_wav, stem_dir_with_all_stems, tmp_path):
        with patch("manual_analyser.audio.separate._run_demucs") as mock_demucs:
            result = separate_track(TRACK_ID, tmp_wav, data_dir=tmp_path, no_cache=False)
        mock_demucs.assert_not_called()
        assert result.device_used == "cached"
        assert isinstance(result, SeparateResult)

    def test_runs_demucs_when_no_cache(self, tmp_wav, stem_dir_with_all_stems, tmp_path):
        with (
            patch("manual_analyser.audio.separate._run_demucs") as mock_demucs,
            patch("manual_analyser.audio.separate.get_torch_device", return_value="cpu"),
        ):
            separate_track(TRACK_ID, tmp_wav, data_dir=tmp_path, no_cache=True)
        mock_demucs.assert_called_once()

    def test_runs_demucs_when_stems_missing(self, tmp_wav, tmp_path):
        def fake_demucs(wav, stem_dir, device, short_id):
            for name in STEM_NAMES:
                (stem_dir / f"{name}.wav").write_bytes(b"RIFF")

        with (
            patch("manual_analyser.audio.separate._run_demucs", side_effect=fake_demucs) as mock_demucs,
            patch("manual_analyser.audio.separate.get_torch_device", return_value="cpu"),
        ):
            separate_track(TRACK_ID, tmp_wav, data_dir=tmp_path)
        mock_demucs.assert_called_once()


# ---------------------------------------------------------------------------
# separate_track — OOM handling
# ---------------------------------------------------------------------------


class TestSeparateTrackOOM:
    def test_retries_on_cpu_when_gpu_oom(self, tmp_wav, tmp_path):
        call_count = {"n": 0}

        def fake_demucs(wav, stem_dir, device, short_id):
            call_count["n"] += 1
            if device != "cpu":
                raise _OOMError_for_test()
            # On CPU, succeed by creating stem files
            for name in STEM_NAMES:
                (stem_dir / f"{name}.wav").write_bytes(b"RIFF")

        with (
            patch("manual_analyser.audio.separate._run_demucs", side_effect=fake_demucs),
            patch("manual_analyser.audio.separate.get_torch_device", return_value="cuda"),
        ):
            result = separate_track(TRACK_ID, tmp_wav, data_dir=tmp_path)

        assert call_count["n"] == 2
        assert result.device_used == "cpu"
        assert result.was_cpu_fallback is True

    def test_raises_skip_when_cpu_also_oom(self, tmp_wav, tmp_path):
        from manual_analyser.audio.separate import _OOMError

        with (
            patch("manual_analyser.audio.separate._run_demucs", side_effect=_OOMError("out of memory")),
            patch("manual_analyser.audio.separate.get_torch_device", return_value="cuda"),
        ):
            with pytest.raises(SeparateSkipError) as exc_info:
                separate_track(TRACK_ID, tmp_wav, data_dir=tmp_path)
        assert "oom" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# separate_track — incomplete stems
# ---------------------------------------------------------------------------


class TestSeparateTrackIncompleteStems:
    def test_raises_skip_when_stems_incomplete(self, tmp_wav, tmp_path):
        def fake_demucs(wav, stem_dir, device, short_id):
            # Only write drums — missing bass, vocals, other
            (stem_dir / "drums.wav").write_bytes(b"RIFF")

        with (
            patch("manual_analyser.audio.separate._run_demucs", side_effect=fake_demucs),
            patch("manual_analyser.audio.separate.get_torch_device", return_value="cpu"),
        ):
            with pytest.raises(SeparateSkipError) as exc_info:
                separate_track(TRACK_ID, tmp_wav, data_dir=tmp_path)
        assert "missing" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# separate_track — successful result
# ---------------------------------------------------------------------------


class TestSeparateTrackResult:
    def test_returns_correct_result_on_success(self, tmp_wav, tmp_path):
        def fake_demucs(wav, stem_dir, device, short_id):
            for name in STEM_NAMES:
                (stem_dir / f"{name}.wav").write_bytes(b"RIFF")

        with (
            patch("manual_analyser.audio.separate._run_demucs", side_effect=fake_demucs),
            patch("manual_analyser.audio.separate.get_torch_device", return_value="mps"),
        ):
            result = separate_track(TRACK_ID, tmp_wav, data_dir=tmp_path)

        assert isinstance(result, SeparateResult)
        assert result.track_id == TRACK_ID
        assert result.device_used == "mps"
        assert result.was_cpu_fallback is False
        assert result.drums.name == "drums.wav"
        assert result.vocals.name == "vocals.wav"
        assert result.drums.exists()
        assert result.bass.exists()
        assert result.vocals.exists()
        assert result.other.exists()
