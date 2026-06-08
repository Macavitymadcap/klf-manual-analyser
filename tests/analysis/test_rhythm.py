"""Tests for manual_analyser.analysis.rhythm"""

import sqlite3
import wave
from unittest.mock import patch

import numpy as np
import pytest

from manual_analyser.analysis.rhythm import (
    NULL_PATTERN,
    PATTERN_STEPS,
    RhythmResult,
    _compute_syncopation,
    _null_result,
    analyse_rhythm,
)
from manual_analyser.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "c" * 32


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a minimal 2-second silence WAV."""
    wav_path = tmp_path / "drums.wav"
    sr = 44100
    samples = np.zeros(sr * 2, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return wav_path


@pytest.fixture
def db_with_track(tmp_path):
    """Create a DB with a minimal tracks row."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO tracks (track_id, filename, duration, analysis_timestamp, analysis_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (TRACK_ID, "test.mp3", 120.0, "2025-01-01T00:00:00+00:00", "0.1.0"),
        )
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# _null_result
# ---------------------------------------------------------------------------


class TestNullResult:
    def test_returns_rhythm_result(self):
        result = _null_result()
        assert isinstance(result, RhythmResult)

    def test_patterns_are_null_pattern(self):
        result = _null_result()
        assert result.kick_pattern == NULL_PATTERN
        assert result.snare_pattern == NULL_PATTERN
        assert result.hihat_pattern == NULL_PATTERN

    def test_groove_feel_is_unclear(self):
        result = _null_result()
        assert result.groove_feel == "unclear"

    def test_patterns_are_16_chars(self):
        result = _null_result()
        assert len(result.kick_pattern) == PATTERN_STEPS


# ---------------------------------------------------------------------------
# _compute_syncopation
# ---------------------------------------------------------------------------


class TestComputeSyncopation:
    def test_empty_onsets_returns_zero(self):
        beats = np.array([0, 100, 200, 300])
        result = _compute_syncopation(np.array([]), beats, sr=44100)
        assert result == pytest.approx(0.0)

    def test_empty_beats_returns_zero(self):
        onsets = np.array([50, 150, 250])
        result = _compute_syncopation(onsets, np.array([]), sr=44100)
        assert result == pytest.approx(0.0)

    def test_all_on_beat_returns_zero(self):
        beats = np.array([0, 100, 200, 300, 400])
        # Onsets exactly on beats
        result = _compute_syncopation(beats.copy(), beats, sr=44100)
        assert result == pytest.approx(0.0)

    def test_all_off_beat_returns_one(self):
        beats = np.array([0, 100, 200, 300, 400])
        # Onsets midway between beats — maximally off-beat
        off_beat_onsets = np.array([50, 150, 250, 350])
        result = _compute_syncopation(off_beat_onsets, beats, sr=44100)
        assert result == pytest.approx(1.0)

    def test_returns_value_in_range(self):
        beats = np.arange(0, 1000, 100)
        onsets = np.arange(10, 1000, 47)  # irregular offsets
        result = _compute_syncopation(onsets, beats, sr=44100)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# analyse_rhythm — happy path
# ---------------------------------------------------------------------------


class TestAnalyseRhythm:
    def _mock_result(self):
        return RhythmResult(
            kick_pattern="1000100010001000",
            snare_pattern="0000100000001000",
            hihat_pattern="1010101010101010",
            syncopation_score=0.25,
            rhythmic_density=0.6,
            groove_feel="straight",
        )

    def test_returns_rhythm_result(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.rhythm._compute_rhythm",
            return_value=self._mock_result(),
        ):
            result = analyse_rhythm(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        assert isinstance(result, RhythmResult)
        assert result.kick_pattern == "1000100010001000"
        assert result.groove_feel == "straight"

    def test_writes_beat_patterns_to_db(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.rhythm._compute_rhythm",
            return_value=self._mock_result(),
        ):
            analyse_rhythm(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM beat_patterns WHERE track_id = ?", (TRACK_ID,)).fetchone()
        conn.close()

        assert row is not None
        assert row["kick_pattern"] == "1000100010001000"
        assert row["snare_pattern"] == "0000100000001000"
        assert row["hihat_pattern"] == "1010101010101010"
        assert row["syncopation_score"] == pytest.approx(0.25)
        assert row["rhythmic_density"] == pytest.approx(0.6)

    def test_writes_groove_feel_to_tracks(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.rhythm._compute_rhythm",
            return_value=self._mock_result(),
        ):
            analyse_rhythm(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT groove_feel FROM tracks WHERE track_id = ?", (TRACK_ID,)).fetchone()
        conn.close()

        assert row["groove_feel"] == "straight"

    def test_returns_none_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.rhythm._compute_rhythm",
            side_effect=Exception("librosa exploded"),
        ):
            result = analyse_rhythm(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None

    def test_does_not_raise_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.rhythm._compute_rhythm",
            side_effect=RuntimeError("unexpected"),
        ):
            result = analyse_rhythm(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None

    def test_writes_null_pattern_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.rhythm._compute_rhythm",
            side_effect=Exception("fail"),
        ):
            analyse_rhythm(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        bp_row = conn.execute(
            "SELECT kick_pattern FROM beat_patterns WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        t_row = conn.execute(
            "SELECT groove_feel FROM tracks WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert bp_row is not None
        assert bp_row["kick_pattern"] == NULL_PATTERN
        assert t_row["groove_feel"] == "unclear"  # _write_nulls sets this on tracks
