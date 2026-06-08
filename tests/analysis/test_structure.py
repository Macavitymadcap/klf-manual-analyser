"""Tests for manual_analyser.analysis.structure"""

import json
import sqlite3
import wave

import numpy as np
import pytest

from manual_analyser.analysis.structure import (
    SectionLabel,
    _assign_labels,
    _compute_lyric_features,
    _compute_section_energies,
    _find_repeated_phrase,
    align_sections,
    segment_track,
)
from manual_analyser.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "g" * 32
SR = 44100


@pytest.fixture
def tmp_wav(tmp_path):
    """4-second sine wave WAV."""
    wav_path = tmp_path / "full.wav"
    t = np.linspace(0, 4, SR * 4)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
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
            (TRACK_ID, "test.mp3", 180.0, "2025-01-01T00:00:00+00:00", "0.1.0"),
        )
    conn.close()
    return db_path


@pytest.fixture
def db_with_sections_and_energy(tmp_path):
    """DB with sections, RMS profile, and transcript segments pre-populated."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO tracks (track_id, filename, duration, analysis_timestamp, analysis_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (TRACK_ID, "test.mp3", 180.0, "2025-01-01T00:00:00+00:00", "0.1.0"),
        )

        # 8 sections of ~22 seconds each
        for i in range(8):
            start = i * 22.5
            end = (i + 1) * 22.5
            conn.execute(
                """
                INSERT INTO sections
                    (track_id, position, start, end, duration, 
                    label, label_confidence, label_source)
                VALUES (?, ?, ?, ?, ?, 'unknown', 0.0, 'acoustic')
                """,
                (TRACK_ID, i, start, end, 22.5),
            )

        # RMS profile: 360 values (180s at 0.5s intervals)
        # Shape: quiet intro, verse, loud chorus, verse, loud chorus,
        #        quiet breakdown, very loud double chorus, quiet outro
        energies = (
            [0.2] * 45  # intro (0-22.5s)
            + [0.5] * 45  # verse
            + [0.9] * 45  # chorus (loud)
            + [0.5] * 45  # verse
            + [0.9] * 45  # chorus (loud)
            + [0.15] * 45  # breakdown (quiet)
            + [0.95] * 45  # double chorus (very loud)
            + [0.2] * 45  # outro
        )
        conn.execute(
            "INSERT INTO tracks_timeseries (track_id, rms_profile_json) VALUES (?, ?)",
            (TRACK_ID, json.dumps(energies)),
        )

        # Transcript: chorus phrase "baby baby yeah" in sections 2, 4, and 6
        for section_idx, phrases in {
            0: [],  # intro — no lyrics
            1: ["gonna tell you", "what I feel"],  # verse
            2: ["baby baby yeah"] * 5,  # chorus — repeated phrase
            3: ["walking down the street"],  # verse
            4: ["baby baby yeah"] * 5,  # chorus — repeated phrase
            5: [],  # breakdown — no lyrics
            6: ["baby baby yeah"] * 8,  # double chorus
            7: [],  # outro — no lyrics
        }.items():
            start_base = section_idx * 22.5
            for j, phrase in enumerate(phrases):
                start = start_base + j * 2.0
                end = start + 1.8
                conn.execute(
                    """INSERT INTO transcript_segments (
                        track_id, 
                        start, 
                        end, 
                        text
                      ) VALUES (?, ?, ?, ?)""",
                    (TRACK_ID, start, end, phrase),
                )

    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# _find_repeated_phrase
# ---------------------------------------------------------------------------


class TestFindRepeatedPhrase:
    def test_finds_repeated_phrase(self):
        words = "baby baby yeah baby baby yeah something else".split()
        result = _find_repeated_phrase(words)
        assert result == "baby baby yeah"

    def test_returns_none_for_no_repetition(self):
        words = "one two three four five six".split()
        result = _find_repeated_phrase(words)
        assert result is None

    def test_returns_none_for_empty(self):
        assert _find_repeated_phrase([]) is None

    def test_returns_none_for_short_words(self):
        words = "one two".split()
        assert _find_repeated_phrase(words) is None


# ---------------------------------------------------------------------------
# _compute_section_energies
# ---------------------------------------------------------------------------


class TestComputeSectionEnergies:
    def test_returns_list_per_section(self):
        sections = [
            {"start": 0.0, "end": 10.0},
            {"start": 10.0, "end": 20.0},
        ]
        rms = np.array([0.3] * 20 + [0.7] * 20)
        result = _compute_section_energies(sections, rms, 20.0)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.3, abs=0.05)
        assert result[1] == pytest.approx(0.7, abs=0.05)

    def test_empty_rms_returns_neutral(self):
        sections = [{"start": 0.0, "end": 10.0}]
        result = _compute_section_energies(sections, np.array([]), 10.0)
        assert result == [0.5]

    def test_returns_values_in_range(self):
        sections = [{"start": i * 10.0, "end": (i + 1) * 10.0} for i in range(5)]
        rms = np.random.default_rng(0).uniform(0, 1, 100)
        result = _compute_section_energies(sections, rms, 50.0)
        assert all(0.0 <= v <= 1.0 for v in result)


# ---------------------------------------------------------------------------
# _compute_lyric_features
# ---------------------------------------------------------------------------


class TestComputeLyricFeatures:
    def test_returns_list_per_section(self):
        sections = [
            {"start": 0.0, "end": 10.0},
            {"start": 10.0, "end": 20.0},
        ]

        class FakeRow:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

            def __getitem__(self, key):
                return getattr(self, key)

        transcript = [
            FakeRow(12.0, 13.0, "baby baby yeah"),
            FakeRow(14.0, 15.0, "baby baby yeah"),
        ]
        result = _compute_lyric_features(sections, transcript)
        assert len(result) == 2
        # Section 0 has no transcript overlap
        assert result[0]["lyric_density"] == pytest.approx(0.0)
        # Section 1 has transcript
        assert result[1]["lyric_density"] > 0.0

    def test_empty_transcript_returns_zero_density(self):
        sections = [{"start": 0.0, "end": 10.0}]
        result = _compute_lyric_features(sections, [])
        assert result[0]["lyric_density"] == 0.0
        assert result[0]["repeated_phrase"] is None


# ---------------------------------------------------------------------------
# _assign_labels
# ---------------------------------------------------------------------------


class TestAssignLabels:
    def _make_sections(self, n, duration=180.0):
        seg_len = duration / n
        return [{"id": i, "pos": i, "start": i * seg_len, "end": (i + 1) * seg_len} for i in range(n)]

    def _make_lyric_data(self, n):
        from collections import Counter

        return [
            {"lyric_density": 0.3, "word_count": 10, "phrases": Counter(), "repeated_phrase": None} for _ in range(n)
        ]

    def test_returns_list_of_section_labels(self):
        n = 8
        sections = self._make_sections(n)
        energies = [0.5] * n
        lyric_data = self._make_lyric_data(n)
        result = _assign_labels(sections, energies, lyric_data, 180.0, "test")
        assert len(result) == n
        assert all(isinstance(s, SectionLabel) for s in result)

    def test_all_labels_are_valid(self):
        valid = {
            "intro",
            "verse",
            "pre_chorus",
            "chorus",
            "breakdown",
            "double_chorus",
            "bridge",
            "outro",
            "unknown",
        }
        n = 8
        sections = self._make_sections(n)
        energies = [0.2, 0.5, 0.9, 0.5, 0.9, 0.15, 0.95, 0.2]
        lyric_data = self._make_lyric_data(n)
        result = _assign_labels(sections, energies, lyric_data, 180.0, "test")
        for s in result:
            assert s.label in valid

    def test_intro_assigned_to_quiet_first_section(self):
        n = 8
        sections = self._make_sections(n)
        energies = [0.1, 0.6, 0.9, 0.6, 0.9, 0.2, 0.9, 0.2]
        from collections import Counter

        lyric_data = [
            {"lyric_density": 0.0, "word_count": 0, "phrases": Counter(), "repeated_phrase": None},
        ] + [{"lyric_density": 0.4, "word_count": 15, "phrases": Counter(), "repeated_phrase": None}] * 7
        result = _assign_labels(sections, energies, lyric_data, 180.0, "test")
        assert result[0].label == "intro"

    def test_confidence_in_range(self):
        n = 8
        sections = self._make_sections(n)
        energies = [0.5] * n
        lyric_data = self._make_lyric_data(n)
        result = _assign_labels(sections, energies, lyric_data, 180.0, "test")
        for s in result:
            assert 0.0 <= s.label_confidence <= 1.0


# ---------------------------------------------------------------------------
# segment_track (pass 1)
# ---------------------------------------------------------------------------


class TestSegmentTrack:
    def test_returns_boundary_list(self, tmp_wav, db_with_track, tmp_path):
        result = segment_track(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_writes_section_skeletons(self, tmp_wav, db_with_track, tmp_path):
        segment_track(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)

        conn = sqlite3.connect(db_with_track)
        count = conn.execute("SELECT COUNT(*) FROM sections WHERE track_id = ?", (TRACK_ID,)).fetchone()[0]
        conn.close()

        assert count >= 1

    def test_does_not_duplicate_sections(self, tmp_wav, db_with_track, tmp_path):
        """Running twice should not add duplicate section rows."""
        segment_track(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        count_after_first = (
            sqlite3.connect(db_with_track)
            .execute("SELECT COUNT(*) FROM sections WHERE track_id = ?", (TRACK_ID,))
            .fetchone()[0]
        )

        segment_track(TRACK_ID, tmp_wav, data_dir=tmp_path, db_path=db_with_track)
        count_after_second = (
            sqlite3.connect(db_with_track)
            .execute("SELECT COUNT(*) FROM sections WHERE track_id = ?", (TRACK_ID,))
            .fetchone()[0]
        )

        assert count_after_first == count_after_second

    def test_returns_empty_on_failure(self, tmp_path, db_with_track):
        bad_wav = tmp_path / "nonexistent.wav"
        result = segment_track(TRACK_ID, bad_wav, data_dir=tmp_path, db_path=db_with_track)
        assert result == []


# ---------------------------------------------------------------------------
# align_sections (pass 2)
# ---------------------------------------------------------------------------


class TestAlignSections:
    def test_returns_list_of_section_labels(self, db_with_sections_and_energy, tmp_path):
        result = align_sections(TRACK_ID, data_dir=tmp_path, db_path=db_with_sections_and_energy)
        assert result is not None
        assert len(result) == 8
        assert all(isinstance(s, SectionLabel) for s in result)

    def test_labels_chorus_sections(self, db_with_sections_and_energy, tmp_path):
        result = align_sections(TRACK_ID, data_dir=tmp_path, db_path=db_with_sections_and_energy)
        labels = [s.label for s in result]
        # Sections 2 and 4 should be chorus (high repetition + high energy)
        assert "chorus" in labels

    def test_labels_breakdown(self, db_with_sections_and_energy, tmp_path):
        result = align_sections(TRACK_ID, data_dir=tmp_path, db_path=db_with_sections_and_energy)
        labels = [s.label for s in result]
        # Section 5 has lowest energy in second half — should be breakdown
        assert "breakdown" in labels

    def test_writes_labels_to_db(self, db_with_sections_and_energy, tmp_path):
        align_sections(TRACK_ID, data_dir=tmp_path, db_path=db_with_sections_and_energy)

        conn = sqlite3.connect(db_with_sections_and_energy)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT label, label_confidence FROM sections WHERE track_id = ? ORDER BY position",
            (TRACK_ID,),
        ).fetchall()
        conn.close()

        assert len(rows) == 8
        # At least some sections should now have labels other than "unknown"
        non_unknown = [r for r in rows if r["label"] != "unknown"]
        assert len(non_unknown) > 0

    def test_returns_none_when_no_sections(self, db_with_track, tmp_path):
        """No sections in DB — should return None gracefully (empty list actually)."""
        result = align_sections(TRACK_ID, data_dir=tmp_path, db_path=db_with_track)
        # Returns empty list (not None) when no sections
        assert result == [] or result is None
