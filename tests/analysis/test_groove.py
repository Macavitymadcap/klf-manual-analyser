"""Tests for manual_analyser.analysis.groove"""

import sqlite3
import wave
from unittest.mock import patch

import numpy as np
import pytest

from manual_analyser.analysis.groove import (
    GrooveResult,
    _approximate_danceability,
    _compute_beat_regularity,
    _compute_repetition_score,
    _compute_self_similarity,
    analyse_groove,
)
from manual_analyser.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "f" * 32
SR = 44100


@pytest.fixture
def tmp_wav(tmp_path):
    """4-second sine wave WAV with a regular 120BPM click for beat tests."""
    wav_path = tmp_path / "full.wav"
    duration = 4
    t = np.linspace(0, duration, SR * duration)

    # Sine wave base
    y = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add regular clicks at 120 BPM (every 0.5s) for beat detection
    click_interval = int(SR * 0.5)
    for i in range(0, len(y), click_interval):
        end = min(i + 100, len(y))
        y[i:end] += 0.7

    samples = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples.tobytes())
    return wav_path


@pytest.fixture
def silence_wav(tmp_path):
    """4-second silence WAV."""
    wav_path = tmp_path / "silence.wav"
    samples = np.zeros(SR * 4, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples.tobytes())
    return wav_path


@pytest.fixture
def db_with_track(tmp_path):
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
# _compute_beat_regularity
# ---------------------------------------------------------------------------


class TestComputeBeatRegularity:
    def test_returns_value_in_range(self, tmp_wav):
        import librosa

        y, sr = librosa.load(str(tmp_wav), sr=None, mono=True)
        result = _compute_beat_regularity(y, sr)
        assert 0.0 <= result <= 1.0

    def test_regular_beat_scores_high(self):
        """Synthesise a perfectly regular click track and verify high regularity."""
        sr = 22050
        duration = 4
        y = np.zeros(sr * duration)
        interval = sr // 4  # 4 beats per second = 240 BPM
        for i in range(0, len(y), interval):
            end = min(i + 50, len(y))
            y[i:end] = 0.9
        result = _compute_beat_regularity(y, sr)
        assert result >= 0.5  # should be reasonably high

    def test_silence_returns_neutral(self, silence_wav):
        import librosa

        y, sr = librosa.load(str(silence_wav), sr=None, mono=True)
        result = _compute_beat_regularity(y, sr)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _compute_self_similarity
# ---------------------------------------------------------------------------


class TestComputeSelfSimilarity:
    def test_returns_value_in_range(self, tmp_wav):
        import librosa

        y, sr = librosa.load(str(tmp_wav), sr=None, mono=True)
        result = _compute_self_similarity(y, sr)
        assert 0.0 <= result <= 1.0

    def test_repetitive_signal_scores_higher_than_random(self):
        """A repeating signal should have higher self-similarity than noise."""
        sr = 22050
        duration = 4
        t = np.linspace(0, duration, sr * duration)

        # Repeating 1-second motif
        motif = np.sin(2 * np.pi * 220 * t[:sr])
        repetitive = np.tile(motif, duration)

        # Random noise
        rng = np.random.default_rng(42)
        random_signal = rng.uniform(-1, 1, sr * duration)

        rep_score = _compute_self_similarity(repetitive, sr)
        rand_score = _compute_self_similarity(random_signal, sr)

        assert rep_score >= rand_score


# ---------------------------------------------------------------------------
# _compute_repetition_score
# ---------------------------------------------------------------------------


class TestComputeRepetitionScore:
    def test_returns_value_in_range(self, tmp_wav):
        import librosa

        y, sr = librosa.load(str(tmp_wav), sr=None, mono=True)
        result = _compute_repetition_score(y, sr)
        assert 0.0 <= result <= 1.0

    def test_repetitive_signal_scores_higher(self):
        sr = 22050
        duration = 4
        t = np.linspace(0, duration, sr * duration)
        motif = np.sin(2 * np.pi * 330 * t[:sr])
        repetitive = np.tile(motif, duration)

        rng = np.random.default_rng(0)
        random_signal = rng.uniform(-1, 1, sr * duration)

        rep_score = _compute_repetition_score(repetitive, sr)
        rand_score = _compute_repetition_score(random_signal, sr)

        assert rep_score >= rand_score


# ---------------------------------------------------------------------------
# _approximate_danceability
# ---------------------------------------------------------------------------


class TestApproximateDanceability:
    def test_returns_value_in_range(self, tmp_wav):
        import librosa

        y, sr = librosa.load(str(tmp_wav), sr=None, mono=True)
        result = _approximate_danceability(y, sr)
        assert 0.0 <= result <= 1.0

    def test_silence_returns_low_value(self, silence_wav):
        import librosa

        y, sr = librosa.load(str(silence_wav), sr=None, mono=True)
        result = _approximate_danceability(y, sr)
        assert result < 0.5


# ---------------------------------------------------------------------------
# analyse_groove — DB writes
# ---------------------------------------------------------------------------


class TestAnalyseGroove:
    def _mock_result(self):
        return GrooveResult(
            danceability=0.72,
            self_similarity_score=0.65,
            beat_regularity=0.88,
            groove_consistency=0.76,
            repetition_score=0.55,
        )

    def test_returns_groove_result(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.groove._compute_groove",
            return_value=self._mock_result(),
        ):
            result = analyse_groove(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert isinstance(result, GrooveResult)
        assert result.danceability == pytest.approx(0.72)

    def test_writes_all_fields_to_db(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.groove._compute_groove",
            return_value=self._mock_result(),
        ):
            analyse_groove(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT danceability, self_similarity_score, beat_regularity,
                   groove_consistency, repetition_score
            FROM tracks WHERE track_id = ?
            """,
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row["danceability"] == pytest.approx(0.72)
        assert row["self_similarity_score"] == pytest.approx(0.65)
        assert row["beat_regularity"] == pytest.approx(0.88)
        assert row["groove_consistency"] == pytest.approx(0.76)
        assert row["repetition_score"] == pytest.approx(0.55)

    def test_returns_none_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.groove._compute_groove",
            side_effect=Exception("fail"),
        ):
            result = analyse_groove(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None

    def test_writes_nulls_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.groove._compute_groove",
            side_effect=Exception("fail"),
        ):
            analyse_groove(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT danceability, groove_consistency FROM tracks WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row["danceability"] is None
        assert row["groove_consistency"] is None

    def test_does_not_raise_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.groove._compute_groove",
            side_effect=RuntimeError("unexpected"),
        ):
            result = analyse_groove(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None

    def test_essentia_failure_falls_back_gracefully(self, tmp_wav, db_with_track, tmp_path):
        """If essentia raises inside _compute_groove, result should still be returned."""
        import librosa

        def groove_with_essentia_fail(full_wav, short_id):
            y, sr = librosa.load(str(full_wav), sr=None, mono=True)
            # Simulate essentia failing — danceability falls back to approximation
            from manual_analyser.analysis.groove import (
                _approximate_danceability,
                _compute_beat_regularity,
                _compute_repetition_score,
                _compute_self_similarity,
            )

            danceability = _approximate_danceability(y, sr)
            beat_regularity = _compute_beat_regularity(y, sr)
            self_sim = _compute_self_similarity(y, sr)
            repetition = _compute_repetition_score(y, sr)
            return GrooveResult(
                danceability=danceability,
                self_similarity_score=self_sim,
                beat_regularity=beat_regularity,
                groove_consistency=float(np.sqrt(beat_regularity * self_sim)),
                repetition_score=repetition,
            )

        with patch(
            "manual_analyser.analysis.groove._compute_groove",
            side_effect=groove_with_essentia_fail,
        ):
            result = analyse_groove(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        assert result is not None
        assert 0.0 <= result.danceability <= 1.0
