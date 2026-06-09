"""Tests for embedding/summarise.py"""

from manual_analyser.embedding.db_reader import TrackFeatures
from manual_analyser.embedding.summarise import build_summary

TRACK_ID = "a" * 32


def _features(**kwargs) -> TrackFeatures:
    defaults = dict(
        track_id=TRACK_ID,
        artist="The KLF",
        song_name="Doctorin The Tardis",
        bpm=126.0,
        key="C",
        mode="major",
        groove_feel="straight",
        energy_shape="building",
        danceability=0.75,
        hook_phrase="doctorin the tardis",
        hook_repetition_count=8,
        unique_word_ratio=0.3,
        section_labels=["intro", "verse", "chorus", "outro"],
        kick_pattern="1000100010001000",
        snare_pattern="0000100000001000",
    )
    defaults.update(kwargs)
    return TrackFeatures(**defaults)


class TestBuildSummary:
    def test_contains_artist_and_title(self):
        result = build_summary(_features())
        assert "The KLF" in result
        assert "Doctorin The Tardis" in result

    def test_contains_bpm(self):
        result = build_summary(_features())
        assert "126.0 BPM" in result

    def test_contains_key_and_mode(self):
        result = build_summary(_features())
        assert "C major" in result

    def test_contains_groove_feel(self):
        result = build_summary(_features())
        assert "groove=straight" in result

    def test_contains_structure(self):
        result = build_summary(_features())
        assert "intro → verse → chorus → outro" in result

    def test_contains_hook(self):
        result = build_summary(_features())
        assert "doctorin the tardis" in result
        assert "x8" in result

    def test_contains_kick_pattern(self):
        result = build_summary(_features())
        assert "1000100010001000" in result

    def test_none_bpm_omitted(self):
        result = build_summary(_features(bpm=None))
        assert "BPM" not in result

    def test_none_key_omitted(self):
        result = build_summary(_features(key=None))
        assert "Key:" not in result

    def test_empty_sections_omitted(self):
        result = build_summary(_features(section_labels=[]))
        assert "Structure:" not in result

    def test_no_hook_phrase_omitted(self):
        result = build_summary(_features(hook_phrase=None))
        assert "hook=" not in result

    def test_unknown_artist_uses_track_id(self):
        result = build_summary(_features(artist=None, song_name=None))
        assert TRACK_ID[:8] in result

    def test_returns_multiline_string(self):
        result = build_summary(_features())
        assert "\n" in result
