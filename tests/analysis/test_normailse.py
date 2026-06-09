"""Tests for manual_analyser.analysis.normalise"""

import pytest

from manual_analyser.analysis.normalise import (
    normalise_dynamic_range,
    normalise_loudness,
    normalise_lyric_density,
    normalise_rhythmic_density,
    normalise_verse_chorus_delta,
)


class TestNormaliseLoudness:
    def test_min(self):
        assert normalise_loudness(-60.0) == pytest.approx(0.0)

    def test_max(self):
        assert normalise_loudness(0.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert normalise_loudness(-30.0) == pytest.approx(0.5)

    def test_clamps_below(self):
        assert normalise_loudness(-80.0) == pytest.approx(0.0)

    def test_clamps_above(self):
        assert normalise_loudness(10.0) == pytest.approx(1.0)


class TestNormaliseDynamicRange:
    def test_min(self):
        assert normalise_dynamic_range(0.0) == pytest.approx(0.0)

    def test_max(self):
        assert normalise_dynamic_range(60.0) == pytest.approx(1.0)

    def test_clamps(self):
        assert normalise_dynamic_range(120.0) == pytest.approx(1.0)


class TestNormaliseVerseChorusDelta:
    def test_3db_equals_0_15(self):
        # Key threshold used in criteria — must stay exact
        assert normalise_verse_chorus_delta(3.0) == pytest.approx(0.15)

    def test_6db_equals_0_30(self):
        assert normalise_verse_chorus_delta(6.0) == pytest.approx(0.30)

    def test_clamps(self):
        assert normalise_verse_chorus_delta(25.0) == pytest.approx(1.0)


class TestNormaliseLyricDensity:
    def test_zero(self):
        assert normalise_lyric_density(0.0) == pytest.approx(0.0)

    def test_max(self):
        assert normalise_lyric_density(5.0) == pytest.approx(1.0)

    def test_clamps(self):
        assert normalise_lyric_density(10.0) == pytest.approx(1.0)


class TestNormaliseRhythmicDensity:
    def test_zero(self):
        assert normalise_rhythmic_density(0.0) == pytest.approx(0.0)

    def test_max(self):
        assert normalise_rhythmic_density(4.0) == pytest.approx(1.0)
