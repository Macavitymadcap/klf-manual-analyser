"""Tests for manual_analyser.analysis.harmony"""

import json
import sqlite3
import wave
from unittest.mock import patch

import numpy as np
import pytest

from manual_analyser.analysis.harmony import (
    ChordEvent,
    HarmonyResult,
    SectionHarmony,
    _chords_to_progression,
    _detect_key,
    _get_section_boundaries,
    _match_chord,
    analyse_harmony,
)
from manual_analyser.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "e" * 32


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a minimal 4-second WAV with a 440Hz tone."""
    wav_path = tmp_path / "full.wav"
    sr = 44100
    t = np.linspace(0, 4, sr * 4)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
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


@pytest.fixture
def db_with_sections(db_with_track):
    """DB with pre-existing section boundaries (as if structure.py ran first)."""
    conn = get_connection(db_with_track)
    with conn:
        for i, (start, end) in enumerate([(0.0, 1.0), (1.0, 2.5), (2.5, 4.0)]):
            conn.execute(
                """
                INSERT INTO sections
                    (track_id, position, start, end, duration, 
                    label, label_confidence, label_source)
                VALUES (?, ?, ?, ?, ?, 'unknown', 0.0, 'acoustic')
                """,
                (TRACK_ID, i, start, end, end - start),
            )
    conn.close()
    return db_with_track


# ---------------------------------------------------------------------------
# _detect_key
# ---------------------------------------------------------------------------


class TestDetectKey:
    def test_returns_valid_key_and_mode(self):
        rng = np.random.default_rng(42)
        chroma = rng.uniform(0, 1, (12, 100))
        key, mode, confidence = _detect_key(chroma)
        assert key in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        assert mode in ("major", "minor")
        assert 0.0 <= confidence <= 1.0

    def test_c_major_chord_detects_c_major(self):
        # Strong C major chroma: C, E, G
        chroma = np.zeros((12, 10))
        chroma[0] = 1.0  # C
        chroma[4] = 0.8  # E
        chroma[7] = 0.9  # G
        key, mode, confidence = _detect_key(chroma)
        assert key == "C"
        assert mode == "major"

    def test_confidence_in_range(self):
        chroma = np.ones((12, 50)) * 0.5
        _, _, confidence = _detect_key(chroma)
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# _match_chord
# ---------------------------------------------------------------------------


class TestMatchChord:
    def test_c_major_frame(self):
        frame = np.zeros(12)
        frame[0] = 1.0  # C
        frame[4] = 1.0  # E
        frame[7] = 1.0  # G
        result = _match_chord(frame)
        assert result == "C"

    def test_a_minor_frame(self):
        frame = np.zeros(12)
        frame[9] = 1.0  # A
        frame[0] = 1.0  # C
        frame[4] = 1.0  # E
        result = _match_chord(frame)
        assert result == "Am"

    def test_silence_returns_c(self):
        frame = np.zeros(12)
        result = _match_chord(frame)
        assert result == "C"

    def test_returns_string(self):
        rng = np.random.default_rng(0)
        frame = rng.uniform(0, 1, 12)
        result = _match_chord(frame)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _chords_to_progression
# ---------------------------------------------------------------------------


class TestChordsToProgression:
    def test_empty_returns_unknown(self):
        assert _chords_to_progression([]) == "unknown"

    def test_deduplicates_consecutive(self):
        chords = [
            ChordEvent(0.0, 1.0, "C"),
            ChordEvent(1.0, 2.0, "C"),  # duplicate
            ChordEvent(2.0, 3.0, "Am"),
            ChordEvent(3.0, 4.0, "G"),
        ]
        result = _chords_to_progression(chords)
        assert result == "C - Am - G"

    def test_single_chord(self):
        chords = [ChordEvent(0.0, 4.0, "Am")]
        assert _chords_to_progression(chords) == "Am"

    def test_limits_to_eight_chords(self):
        chords = [ChordEvent(float(i), float(i + 1), f"C{i}") for i in range(12)]
        result = _chords_to_progression(chords)
        assert len(result.split(" - ")) <= 8


# ---------------------------------------------------------------------------
# _get_section_boundaries
# ---------------------------------------------------------------------------


class TestGetSectionBoundaries:
    def test_returns_fallback_when_no_sections(self, db_with_track, tmp_path):
        boundaries = _get_section_boundaries(db_with_track, TRACK_ID, 8.0, n_fallback=4)
        assert len(boundaries) == 4
        assert boundaries[0] == (0.0, 2.0)
        assert boundaries[-1][1] == pytest.approx(8.0)

    def test_returns_db_sections_when_exist(self, db_with_sections):
        boundaries = _get_section_boundaries(db_with_sections, TRACK_ID, 4.0)
        assert len(boundaries) == 3
        assert boundaries[0] == (0.0, 1.0)
        assert boundaries[1] == (1.0, 2.5)
        assert boundaries[2] == (2.5, 4.0)

    def test_fallback_segments_cover_full_duration(self, db_with_track):
        boundaries = _get_section_boundaries(db_with_track, TRACK_ID, 10.0, n_fallback=5)
        assert boundaries[0][0] == pytest.approx(0.0)
        assert boundaries[-1][1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# analyse_harmony — DB writes
# ---------------------------------------------------------------------------


class TestAnalyseHarmony:
    def _mock_result(self):
        return HarmonyResult(
            key="A",
            mode="minor",
            key_confidence=0.78,
            sections=[
                SectionHarmony(
                    section_id=-1,
                    position=0,
                    start=0.0,
                    end=2.0,
                    progression="Am - G - F - C",
                    chords=[
                        ChordEvent(0.0, 1.0, "Am"),
                        ChordEvent(1.0, 2.0, "G"),
                    ],
                ),
            ],
        )

    def test_returns_harmony_result(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.harmony._compute_harmony",
            return_value=self._mock_result(),
        ):
            result = analyse_harmony(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert isinstance(result, HarmonyResult)
        assert result.key == "A"
        assert result.mode == "minor"

    def test_writes_key_to_tracks(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.harmony._compute_harmony",
            return_value=self._mock_result(),
        ):
            analyse_harmony(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT key, mode, key_confidence FROM tracks WHERE track_id = ?",
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row["key"] == "A"
        assert row["mode"] == "minor"
        assert row["key_confidence"] == pytest.approx(0.78)

    def test_writes_section_skeleton(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.harmony._compute_harmony",
            return_value=self._mock_result(),
        ):
            analyse_harmony(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sections WHERE track_id = ? ORDER BY position",
            (TRACK_ID,),
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["label"] == "unknown"
        assert rows[0]["start"] == pytest.approx(0.0)
        assert rows[0]["end"] == pytest.approx(2.0)

    def test_writes_chord_progressions(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.harmony._compute_harmony",
            return_value=self._mock_result(),
        ):
            analyse_harmony(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT cp.progression, cp.chords_json
            FROM chord_progressions cp
            JOIN sections s ON cp.section_id = s.id
            WHERE s.track_id = ?
            """,
            (TRACK_ID,),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["progression"] == "Am - G - F - C"
        chords = json.loads(row["chords_json"])
        assert len(chords) == 2
        assert chords[0]["chord"] == "Am"

    def test_does_not_duplicate_sections_when_they_exist(self, tmp_wav, db_with_sections, tmp_path):
        """If structure.py already wrote sections, harmony.py should not add more."""
        with patch(
            "manual_analyser.analysis.harmony._compute_harmony",
            return_value=self._mock_result(),
        ):
            analyse_harmony(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_sections)

        conn = sqlite3.connect(db_with_sections)
        count = conn.execute("SELECT COUNT(*) FROM sections WHERE track_id = ?", (TRACK_ID,)).fetchone()[0]
        conn.close()

        # Should still be 3 (from db_with_sections fixture), not 4
        assert count == 3

    def test_returns_none_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.harmony._compute_harmony",
            side_effect=Exception("fail"),
        ):
            result = analyse_harmony(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None

    def test_does_not_raise_on_failure(self, tmp_wav, db_with_track, tmp_path):
        with patch(
            "manual_analyser.analysis.harmony._compute_harmony",
            side_effect=RuntimeError("unexpected"),
        ):
            result = analyse_harmony(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result is None
