"""Tests for manual_analyser.analysis.energy"""

import json
import sqlite3
import wave
from unittest.mock import patch

import numpy as np
import pytest

from manual_analyser.analysis.energy import (
    EnergyResult,
    _classify_energy_shape,
    _compute_dynamic_range,
    _delta_from_halves,
    analyse_energy,
)
from manual_analyser.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "d" * 32


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a minimal 4-second WAV with a slight energy increase."""
    wav_path = tmp_path / "full.wav"
    sr = 44100
    t = np.linspace(0, 4, sr * 4)
    # Sine wave with increasing amplitude — energy "builds"
    amplitude = np.linspace(0.1, 0.9, len(t))
    samples = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
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
            (TRACK_ID, "test.mp3", 4.0, "2025-01-01T00:00:00+00:00", "0.1.0"),
        )
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# _classify_energy_shape
# ---------------------------------------------------------------------------


class TestClassifyEnergyShape:
    def test_too_short_returns_unclear(self):
        assert _classify_energy_shape(np.array([0.5, 0.6, 0.7])) == "unclear"

    def test_flat_signal(self):
        rms = np.ones(100) * 0.5
        assert _classify_energy_shape(rms) == "flat"

    def test_building_signal(self):
        rms = np.linspace(0.1, 0.9, 100)
        result = _classify_energy_shape(rms)
        assert result == "building"

    def test_peaked_signal(self):
        x = np.linspace(0, 1, 100)
        rms = -((x - 0.5) ** 2) + 0.5  # inverted parabola
        rms = rms / rms.max()
        result = _classify_energy_shape(rms)
        assert result == "peaked"

    def test_returns_valid_value(self):
        rms = np.random.default_rng(42).uniform(0, 1, 50)
        result = _classify_energy_shape(rms)
        assert result in ("building", "flat", "peaked", "unclear")


# ---------------------------------------------------------------------------
# _delta_from_halves
# ---------------------------------------------------------------------------


class TestDeltaFromHalves:
    def test_returns_zero_for_short_array(self):
        assert _delta_from_halves(np.array([0.5, 0.5])) == pytest.approx(0.0)

    def test_returns_zero_when_equal_energy(self):
        rms = np.ones(100) * 0.5
        result = _delta_from_halves(rms)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_returns_positive_when_q3_louder(self):
        rms = np.concatenate(
            [
                np.ones(25) * 0.2,  # Q1 — quiet verse proxy
                np.ones(25) * 0.5,  # Q2
                np.ones(25) * 0.8,  # Q3 — loud chorus proxy
                np.ones(25) * 0.7,  # Q4
            ]
        )
        result = _delta_from_halves(rms)
        assert result > 0.0

    def test_returns_value_in_range(self):
        rms = np.random.default_rng(0).uniform(0, 1, 80)
        result = _delta_from_halves(rms)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _compute_dynamic_range
# ---------------------------------------------------------------------------


class TestComputeDynamicRange:
    def test_silence_returns_low_range(self, tmp_wav):
        import librosa

        y, sr = librosa.load(str(tmp_wav), sr=None, mono=True)
        # Silence — near zero dynamic range
        y_silence = np.zeros_like(y)
        result = _compute_dynamic_range(y_silence, sr)
        assert result == pytest.approx(0.0, abs=0.05)

    def test_returns_value_in_range(self, tmp_wav):
        import librosa

        y, sr = librosa.load(str(tmp_wav), sr=None, mono=True)
        result = _compute_dynamic_range(y, sr)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# analyse_energy — happy path
# ---------------------------------------------------------------------------


class TestAnalyseEnergy:
    def _mock_result(self):
        return EnergyResult(
            loudness_db=0.6,
            dynamic_range_db=0.4,
            verse_chorus_delta=0.2,
            energy_shape="building",
            rms_profile=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        )

    def test_returns_energy_result(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.energy._compute_energy",
            return_value=self._mock_result(),
        ):
            result = analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        assert isinstance(result, EnergyResult)
        assert result.energy_shape == "building"
        assert result.loudness_db == pytest.approx(0.6)

    def test_writes_scalars_to_tracks(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.energy._compute_energy",
            return_value=self._mock_result(),
        ):
            analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT loudness_db, dynamic_range_db, verse_chorus_delta, energy_shape
            FROM tracks WHERE track_id = ?
            """,
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row["loudness_db"] == pytest.approx(0.6)
        assert row["dynamic_range_db"] == pytest.approx(0.4)
        assert row["verse_chorus_delta"] == pytest.approx(0.2)
        assert row["energy_shape"] == "building"

    def test_writes_rms_profile_to_timeseries(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.energy._compute_energy",
            return_value=self._mock_result(),
        ):
            analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT rms_profile_json FROM tracks_timeseries WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row is not None
        profile = json.loads(row["rms_profile_json"])
        assert isinstance(profile, list)
        assert profile == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], abs=0.001)

    def test_returns_none_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.energy._compute_energy",
            side_effect=Exception("librosa exploded"),
        ):
            result = analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None

    def test_writes_nulls_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.energy._compute_energy",
            side_effect=Exception("fail"),
        ):
            analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT loudness_db, energy_shape FROM tracks WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row["loudness_db"] is None
        assert row["energy_shape"] is None

    def test_does_not_raise_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.energy._compute_energy",
            side_effect=RuntimeError("unexpected"),
        ):
            result = analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None

    def test_rms_profile_is_idempotent_on_rerun(self, tmp_wav, db_with_track, tmp_path):
        """Running twice should use INSERT OR REPLACE — no duplicate row."""
        with patch(
            "manual_analyser.analysis.energy._compute_energy",
            return_value=self._mock_result(),
        ):
            analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
            analyse_energy(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        count = conn.execute(
            "SELECT COUNT(*) FROM tracks_timeseries WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()[0]
        conn.close()

        assert count == 1
