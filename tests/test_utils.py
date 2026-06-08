"""Tests for manual_analyser.utils"""

from pathlib import Path

import numpy as np
import pytest

from manual_analyser.utils import (
    classify_groove_feel,
    get_torch_device,
    make_track_id,
    normalise_dynamic_range,
    normalise_loudness,
    normalise_lyric_density,
    normalise_rhythmic_density,
    normalise_verse_chorus_delta,
    onsets_to_pattern,
    parse_filename,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# get_torch_device
# ---------------------------------------------------------------------------


class TestGetTorchDevice:
    def test_returns_string(self):
        result = get_torch_device()
        assert isinstance(result, str)

    def test_valid_device(self):
        result = get_torch_device()
        assert result in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# make_track_id
# ---------------------------------------------------------------------------


class TestMakeTrackId:
    def test_returns_32_char_hex(self):
        result = make_track_id("/some/path/track.mp3")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_path_same_id(self):
        path = "/some/path/track.mp3"
        assert make_track_id(path) == make_track_id(path)

    def test_different_paths_different_ids(self):
        assert make_track_id("/path/a.mp3") != make_track_id("/path/b.mp3")

    def test_accepts_path_object(self):
        result = make_track_id(Path("/some/path/track.mp3"))
        assert len(result) == 32


# ---------------------------------------------------------------------------
# parse_filename
# ---------------------------------------------------------------------------


class TestParseFilename:
    def test_standard_format(self):
        artist, title = parse_filename("The_KLF-Doctorin_The_Tardis.mp3")
        assert artist == "The Klf"
        assert title == "Doctorin The Tardis"

    def test_single_word_artist_and_title(self):
        artist, title = parse_filename("Prince-Purple.mp3")
        assert artist == "Prince"
        assert title == "Purple"

    def test_multi_word_both(self):
        artist, title = parse_filename("Louis_Armstrong-Heebie_Jeebies.mp3")
        assert artist == "Louis Armstrong"
        assert title == "Heebie Jeebies"

    def test_hyphen_in_title_preserved(self):
        # Split on first hyphen only — hyphens in title are kept
        artist, title = parse_filename("The_Beatles-A_Day_In_The_Life.mp3")
        assert artist == "The Beatles"
        assert title == "A Day In The Life"

    def test_non_conformant_returns_none(self):
        artist, title = parse_filename("nodash_noformat.mp3")
        assert artist is None
        assert title is None

    def test_accepts_path_object(self):
        artist, title = parse_filename(Path("The_KLF-Doctorin_The_Tardis.mp3"))
        assert artist == "The Klf"
        assert title == "Doctorin The Tardis"

    def test_path_with_directories(self):
        artist, title = parse_filename("/data/input/Louis_Armstrong-Heebie_Jeebies.mp3")
        assert artist == "Louis Armstrong"
        assert title == "Heebie Jeebies"


# ---------------------------------------------------------------------------
# Normalisation functions
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_loudness_min(self):
        assert normalise_loudness(-60.0) == pytest.approx(0.0)

    def test_loudness_max(self):
        assert normalise_loudness(0.0) == pytest.approx(1.0)

    def test_loudness_midpoint(self):
        assert normalise_loudness(-30.0) == pytest.approx(0.5)

    def test_loudness_clamps_below(self):
        assert normalise_loudness(-80.0) == pytest.approx(0.0)

    def test_loudness_clamps_above(self):
        assert normalise_loudness(10.0) == pytest.approx(1.0)

    def test_dynamic_range_min(self):
        assert normalise_dynamic_range(0.0) == pytest.approx(0.0)

    def test_dynamic_range_max(self):
        assert normalise_dynamic_range(60.0) == pytest.approx(1.0)

    def test_dynamic_range_clamps(self):
        assert normalise_dynamic_range(120.0) == pytest.approx(1.0)

    def test_verse_chorus_delta_3db(self):
        # 3dB / 20dB = 0.15 — key threshold in criteria
        assert normalise_verse_chorus_delta(3.0) == pytest.approx(0.15)

    def test_verse_chorus_delta_6db(self):
        assert normalise_verse_chorus_delta(6.0) == pytest.approx(0.30)

    def test_verse_chorus_delta_clamps(self):
        assert normalise_verse_chorus_delta(25.0) == pytest.approx(1.0)

    def test_lyric_density_zero(self):
        assert normalise_lyric_density(0.0) == pytest.approx(0.0)

    def test_lyric_density_max(self):
        assert normalise_lyric_density(5.0) == pytest.approx(1.0)

    def test_lyric_density_clamps(self):
        assert normalise_lyric_density(10.0) == pytest.approx(1.0)

    def test_rhythmic_density_zero(self):
        assert normalise_rhythmic_density(0.0) == pytest.approx(0.0)

    def test_rhythmic_density_max(self):
        assert normalise_rhythmic_density(4.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# utc_now_iso
# ---------------------------------------------------------------------------


class TestUtcNowIso:
    def test_returns_string(self):
        assert isinstance(utc_now_iso(), str)

    def test_contains_timezone_indicator(self):
        result = utc_now_iso()
        assert "+" in result or result.endswith("Z") or "+00:00" in result


# ---------------------------------------------------------------------------
# onsets_to_pattern
# ---------------------------------------------------------------------------


class TestOnsetsToPattern:
    def test_empty_onsets_returns_zeros(self):
        result = onsets_to_pattern(np.array([]), np.array([0, 100, 200, 300]))
        assert result == "0" * 16
        assert len(result) == 16

    def test_empty_beats_returns_zeros(self):
        result = onsets_to_pattern(np.array([10, 50]), np.array([]))
        assert result == "0" * 16

    def test_returns_16_char_string(self):
        beats = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800])
        onsets = np.array([0, 100, 200, 300])
        result = onsets_to_pattern(onsets, beats)
        assert len(result) == 16
        assert all(c in "01" for c in result)

    def test_four_on_the_floor_pattern(self):
        # Beats at 0, 100, 200, 300 frames — onsets on every beat
        # Should produce hits on steps 0, 4, 8, 12 (every beat in 16-step grid)
        beats = np.arange(0, 1600, 100)  # 16 beats
        onsets = np.arange(0, 1600, 100)  # onset on every beat
        result = onsets_to_pattern(onsets, beats)
        assert len(result) == 16
        # First step should be a hit
        assert result[0] == "1"


# ---------------------------------------------------------------------------
# classify_groove_feel
# ---------------------------------------------------------------------------


class TestClassifyGrooveFeel:
    def test_too_few_beats_returns_unclear(self):
        result = classify_groove_feel(np.array([0.0, 0.5, 1.0]), sr=22050)
        assert result == "unclear"

    def test_returns_valid_value(self):
        # Regular beats — should be straight or unclear, not swung
        beat_times = np.arange(0, 10, 0.5)  # beats every 0.5 seconds
        result = classify_groove_feel(beat_times, sr=22050)
        assert result in ("straight", "swung", "unclear")

    def test_perfectly_regular_is_straight(self):
        # Perfectly metronomic beats should classify as straight
        beat_times = np.arange(0, 20, 0.5)
        result = classify_groove_feel(beat_times, sr=22050)
        assert result in ("straight", "unclear")  # perfectly regular, never swung
