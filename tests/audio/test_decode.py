"""Tests for manual_analyser.audio.decode"""

import shutil
import sqlite3
import wave
from unittest.mock import patch

import numpy as np
import pytest

from manual_analyser.audio.decode import (
    DecodeAbortError,
    DecodeResult,
    DecodeSkipError,
    _duration_from_wav_header,
    check_ffmpeg,
    decode_track,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_mp3(tmp_path):
    """Create a dummy .mp3 file (content irrelevant for path/mock tests)."""
    mp3 = tmp_path / "The_KLF-Doctorin_The_Tardis.mp3"
    mp3.write_bytes(b"ID3" + b"\x00" * 100)
    return mp3


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a real 3-second mono WAV file for duration tests."""
    wav_path = tmp_path / "test.wav"
    sample_rate = 44100
    n_frames = sample_rate * 3
    samples = np.zeros(n_frames, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return wav_path


def _place_wav(tmp_wav, tmp_path, track_id_mock):
    """Copy tmp_wav into the expected stem directory location."""
    stem_dir = tmp_path / "stems" / track_id_mock
    stem_dir.mkdir(parents=True, exist_ok=True)
    full_wav = stem_dir / "full.wav"
    shutil.copy(tmp_wav, full_wav)
    return stem_dir, full_wav


# ---------------------------------------------------------------------------
# check_ffmpeg
# ---------------------------------------------------------------------------


class TestCheckFfmpeg:
    def test_raises_abort_when_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(DecodeAbortError) as exc_info:
                check_ffmpeg()
            assert "ffmpeg" in str(exc_info.value).lower()

    def test_passes_when_present(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            check_ffmpeg()  # should not raise


# ---------------------------------------------------------------------------
# decode_track — input validation
# ---------------------------------------------------------------------------


class TestDecodeTrackValidation:
    def test_raises_skip_for_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.mp3"
        with pytest.raises(DecodeSkipError) as exc_info:
            decode_track(missing)
        assert "not found" in str(exc_info.value).lower()

    def test_raises_skip_for_wrong_extension(self, tmp_path):
        bad = tmp_path / "track.wav"
        bad.write_bytes(b"\x00" * 10)
        with pytest.raises(DecodeSkipError) as exc_info:
            decode_track(bad)
        assert ".mp3" in str(exc_info.value)

    def test_raises_abort_when_ffmpeg_missing(self, tmp_mp3, tmp_path):
        with patch("shutil.which", return_value=None):
            with pytest.raises(DecodeAbortError):
                decode_track(tmp_mp3, data_dir=tmp_path)


# ---------------------------------------------------------------------------
# decode_track — caching behaviour
# ---------------------------------------------------------------------------


class TestDecodeTrackCaching:
    def test_skips_ffmpeg_when_wav_exists(self, tmp_mp3, tmp_wav, tmp_path):
        track_id_mock = "a" * 32
        stem_dir, full_wav = _place_wav(tmp_wav, tmp_path, track_id_mock)

        with (
            patch("manual_analyser.audio.decode.make_track_id", return_value=track_id_mock),
            patch("manual_analyser.audio.decode._run_ffmpeg") as mock_ffmpeg,
            patch("manual_analyser.audio.decode._get_duration", return_value=3.0),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("manual_analyser.audio.decode._write_track_row"),
        ):
            result = decode_track(tmp_mp3, data_dir=tmp_path, no_cache=False)

        mock_ffmpeg.assert_not_called()
        assert result.track_id == track_id_mock

    def test_runs_ffmpeg_when_no_cache(self, tmp_mp3, tmp_wav, tmp_path):
        track_id_mock = "b" * 32
        _place_wav(tmp_wav, tmp_path, track_id_mock)

        with (
            patch("manual_analyser.audio.decode.make_track_id", return_value=track_id_mock),
            patch("manual_analyser.audio.decode._run_ffmpeg") as mock_ffmpeg,
            patch("manual_analyser.audio.decode._get_duration", return_value=3.0),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("manual_analyser.audio.decode._write_track_row"),
        ):
            decode_track(tmp_mp3, data_dir=tmp_path, no_cache=True)

        mock_ffmpeg.assert_called_once()

    def test_runs_ffmpeg_when_wav_missing(self, tmp_mp3, tmp_path):
        track_id_mock = "c" * 32

        with (
            patch("manual_analyser.audio.decode.make_track_id", return_value=track_id_mock),
            patch("manual_analyser.audio.decode._run_ffmpeg") as mock_ffmpeg,
            patch("manual_analyser.audio.decode._get_duration", return_value=180.0),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("manual_analyser.audio.decode._write_track_row"),
        ):
            decode_track(tmp_mp3, data_dir=tmp_path)

        mock_ffmpeg.assert_called_once()


# ---------------------------------------------------------------------------
# decode_track — return value
# ---------------------------------------------------------------------------


class TestDecodeTrackResult:
    def test_returns_decode_result(self, tmp_mp3, tmp_wav, tmp_path):
        track_id_mock = "d" * 32
        stem_dir, full_wav = _place_wav(tmp_wav, tmp_path, track_id_mock)

        with (
            patch("manual_analyser.audio.decode.make_track_id", return_value=track_id_mock),
            patch("manual_analyser.audio.decode._get_duration", return_value=3.0),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("manual_analyser.audio.decode._write_track_row"),
        ):
            result = decode_track(tmp_mp3, data_dir=tmp_path)

        assert isinstance(result, DecodeResult)
        assert result.track_id == track_id_mock
        assert result.filename == "The_KLF-Doctorin_The_Tardis.mp3"
        assert result.artist == "The Klf"
        assert result.song_name == "Doctorin The Tardis"
        assert result.duration == pytest.approx(3.0)
        assert result.full_wav == full_wav
        assert result.stem_dir == stem_dir

    def test_non_conformant_filename_stores_null_artist(self, tmp_path, tmp_wav):
        bad_mp3 = tmp_path / "nodash_format.mp3"
        bad_mp3.write_bytes(b"ID3" + b"\x00" * 100)

        track_id_mock = "e" * 32
        _place_wav(tmp_wav, tmp_path, track_id_mock)

        with (
            patch("manual_analyser.audio.decode.make_track_id", return_value=track_id_mock),
            patch("manual_analyser.audio.decode._get_duration", return_value=3.0),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("manual_analyser.audio.decode._write_track_row"),
        ):
            result = decode_track(bad_mp3, data_dir=tmp_path)

        assert result.artist is None
        assert result.song_name is None


# ---------------------------------------------------------------------------
# _duration_from_wav_header
# ---------------------------------------------------------------------------


class TestDurationFromWavHeader:
    def test_reads_duration_correctly(self, tmp_wav):
        duration = _duration_from_wav_header(tmp_wav)
        assert duration == pytest.approx(3.0, abs=0.01)

    def test_returns_zero_for_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a wav file")
        result = _duration_from_wav_header(bad)
        assert result == 0.0


# ---------------------------------------------------------------------------
# SQLite integration
# ---------------------------------------------------------------------------


class TestSQLiteWrite:
    def test_writes_track_row(self, tmp_mp3, tmp_wav, tmp_path):
        track_id_mock = "f" * 32
        _place_wav(tmp_wav, tmp_path, track_id_mock)
        db_path = tmp_path / "test.db"

        with (
            patch("manual_analyser.audio.decode.make_track_id", return_value=track_id_mock),
            patch("manual_analyser.audio.decode._get_duration", return_value=3.0),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        ):
            decode_track(tmp_mp3, data_dir=tmp_path, db_path=db_path)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id_mock,)).fetchone()
        conn.close()

        assert row is not None
        assert row["filename"] == "The_KLF-Doctorin_The_Tardis.mp3"
        assert row["artist"] == "The Klf"
        assert row["song_name"] == "Doctorin The Tardis"
        assert row["duration"] == pytest.approx(3.0)

    def test_insert_ignore_does_not_overwrite(self, tmp_mp3, tmp_wav, tmp_path):
        track_id_mock = "a" * 32
        _place_wav(tmp_wav, tmp_path, track_id_mock)
        db_path = tmp_path / "test.db"

        with (
            patch("manual_analyser.audio.decode.make_track_id", return_value=track_id_mock),
            patch("manual_analyser.audio.decode._get_duration", return_value=3.0),
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        ):
            decode_track(tmp_mp3, data_dir=tmp_path, db_path=db_path)
            decode_track(tmp_mp3, data_dir=tmp_path, db_path=db_path)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM tracks WHERE track_id = ?", (track_id_mock,)).fetchone()[0]
        conn.close()

        assert count == 1
