"""Tests for manual_analyser.analysis.tempo"""

import sqlite3
import wave
from unittest.mock import patch

import numpy as np
import pytest

from manual_analyser.analysis.tempo import (
    TempoResult,
    _compute_tempo_stability,
    _estimate_time_signature,
    analyse_tempo,
)
from manual_analyser.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "b" * 32


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a minimal real WAV file (2 seconds, silence)."""
    wav_path = tmp_path / "full.wav"
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
    """Create a DB with a minimal tracks row ready to be updated."""
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
# _compute_tempo_stability
# ---------------------------------------------------------------------------


class TestComputeTempoStability:
    def test_perfect_regularity_returns_one(self):
        # Perfectly even beats every 0.5 seconds
        beat_times = np.arange(0, 10, 0.5)
        result = _compute_tempo_stability(beat_times)
        assert result == pytest.approx(1.0)

    def test_too_few_beats_returns_neutral(self):
        result = _compute_tempo_stability(np.array([0.0, 0.5]))
        assert result == pytest.approx(0.5)

    def test_high_variance_returns_low_stability(self):
        # Wildly irregular beats
        rng = np.random.default_rng(42)
        beat_times = np.cumsum(rng.uniform(0.3, 1.5, 20))
        result = _compute_tempo_stability(beat_times)
        assert result < 0.5

    def test_returns_value_in_range(self):
        beat_times = np.arange(0, 10, 0.5) + np.random.default_rng(0).normal(0, 0.01, 20)
        result = _compute_tempo_stability(beat_times)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _estimate_time_signature
# ---------------------------------------------------------------------------


class TestEstimateTimeSignature:
    def test_too_few_beats_returns_four(self):
        result = _estimate_time_signature(np.array([0.0, 0.5, 1.0]))
        assert result == 4

    def test_returns_3_or_4(self):
        beat_times = np.arange(0, 10, 0.5)
        result = _estimate_time_signature(beat_times)
        assert result in (3, 4)

    def test_regular_common_time_returns_four(self):
        # Uniform beats in 4/4 — should return 4
        beat_times = np.arange(0, 20, 0.5)
        result = _estimate_time_signature(beat_times)
        assert result == 4


# ---------------------------------------------------------------------------
# analyse_tempo — happy path
# ---------------------------------------------------------------------------


class TestAnalyseTempo:
    def test_returns_tempo_result(self, tmp_wav, db_with_track, tmp_path):
        mock_result = TempoResult(
            bpm=120.0,
            bpm_confidence=0.85,
            time_signature=4,
            tempo_stability=0.92,
            beat_times=np.arange(0, 10, 0.5),
        )

        with patch(
            "manual_analyser.analysis.tempo._compute_tempo",
            return_value=mock_result,
        ):
            result = analyse_tempo(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        assert result is not None
        assert result.bpm == pytest.approx(120.0)
        assert result.time_signature == 4

    def test_writes_bpm_to_db(self, tmp_wav, db_with_track, tmp_path):
        mock_result = TempoResult(
            bpm=127.5,
            bpm_confidence=0.9,
            time_signature=4,
            tempo_stability=0.88,
            beat_times=np.arange(0, 10, 0.47),
        )

        with patch(
            "manual_analyser.analysis.tempo._compute_tempo",
            return_value=mock_result,
        ):
            analyse_tempo(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT bpm, bpm_confidence, time_signature, tempo_stability FROM tracks WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row["bpm"] == pytest.approx(127.5)
        assert row["bpm_confidence"] == pytest.approx(0.9)
        assert row["time_signature"] == 4
        assert row["tempo_stability"] == pytest.approx(0.88)

    def test_returns_none_and_writes_nulls_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.tempo._compute_tempo",
            side_effect=Exception("librosa exploded"),
        ):
            result = analyse_tempo(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        assert result is None

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT bpm, bpm_confidence FROM tracks WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row["bpm"] is None
        assert row["bpm_confidence"] is None

    def test_does_not_raise_on_failure(self, tmp_wav, db_with_track, tmp_path):
        """Analysis failure must not propagate — returns None instead."""
        with patch(
            "manual_analyser.analysis.tempo._compute_tempo",
            side_effect=RuntimeError("unexpected"),
        ):
            result = analyse_tempo(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None
