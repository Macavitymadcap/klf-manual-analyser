"""Tests for manual_analyser.scoring.prompt"""

import pytest

from manual_analyser.db import get_connection
from manual_analyser.scoring.criteria import Criterion, ModeConfig
from manual_analyser.scoring.prompt import (
    PromptPackage,
    _assemble_user_prompt,
    _fetch_field_values,
    _format_value,
    build_all_llm_prompts,
    build_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ID = "j" * 32


@pytest.fixture
def db_with_full_track(tmp_path):
    """DB with a tracks row, section, beat_pattern, and chord_progression."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO tracks (
                track_id, filename, artist, song_name, duration,
                bpm, bpm_confidence, danceability, self_similarity_score,
                beat_regularity, groove_consistency, repetition_score,
                groove_feel, key, mode, key_confidence,
                unique_word_ratio, hook_phrase, hook_repetition_count,
                hook_first_appearance, verse_chorus_delta, energy_shape,
                analysis_timestamp, analysis_version
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )
            """,
            (
                TRACK_ID,
                "The_KLF-Doctorin.mp3",
                "The KLF",
                "Doctorin The Tardis",
                180.0,
                126.0,
                0.88,
                0.72,
                0.65,
                0.84,
                0.74,
                0.58,
                "straight",
                "C",
                "major",
                0.81,
                0.28,
                "doctorin the tardis",
                8,
                12.5,
                0.22,
                "peaked",
                "2025-01-01T00:00:00+00:00",
                "0.1.0",
            ),
        )
        # Section
        conn.execute(
            """
            INSERT INTO sections
                (track_id, position, start, end, duration, label, label_confidence, label_source)
            VALUES (?, 0, 0.0, 20.0, 20.0, 'intro', 0.85, 'acoustic')
            """,
            (TRACK_ID,),
        )
        conn.execute(
            """
            INSERT INTO sections
                (track_id, position, start, end, duration, label, label_confidence, label_source)
            VALUES (?, 1, 20.0, 60.0, 40.0, 'chorus', 0.90, 'hybrid')
            """,
            (TRACK_ID,),
        )
        # Beat pattern
        conn.execute(
            """
            INSERT INTO beat_patterns
                (track_id, kick_pattern, snare_pattern, hihat_pattern,
                 syncopation_score, rhythmic_density)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (TRACK_ID, "1000100010001000", "0000100000001000", "1010101010101010", 0.25, 0.6),
        )
        # Chord progression
        section_id = conn.execute(
            "SELECT id FROM sections WHERE track_id = ? AND position = 1",
            (TRACK_ID,),
        ).fetchone()["id"]
        conn.execute(
            """
            INSERT INTO chord_progressions (section_id, progression, chords_json)
            VALUES (?, ?, ?)
            """,
            (section_id, "C - G - Am - F", "[]"),
        )
    conn.close()
    return db_path


def _make_groove_criterion():
    return Criterion(
        id="groove",
        name="Continuous Dance Groove",
        description="A single groove must run all the way through the record.",
        weight=2.0,
        rule="llm",
        db_fields=[
            "tracks.self_similarity_score",
            "tracks.beat_regularity",
            "tracks.danceability",
            "tracks.groove_consistency",
        ],
        prompt_hint="Evaluate the groove consistency. High scores for all fields = strong groove.",
    )


def _make_mode_config():
    return ModeConfig(
        name="1988",
        description="Test mode",
        system_prompt="You are a test scorer. Respond with JSON only.",
        criteria=[_make_groove_criterion()],
    )


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_float_rounded_to_3dp(self):
        result = _format_value("tracks.bpm", 126.4567)
        assert result == "126.457"

    def test_beat_pattern_grouped(self):
        result = _format_value("tracks.kick_pattern", "1000100010001000")
        assert result == "1000 1000 1000 1000"

    def test_long_string_truncated(self):
        long_val = "a" * 600
        result = _format_value("tracks.hook_phrase", long_val)
        assert len(result) <= 503
        assert result.endswith("...")

    def test_integer_as_string(self):
        result = _format_value("tracks.hook_repetition_count", 8)
        assert result == "8"

    def test_short_string_unchanged(self):
        result = _format_value("tracks.groove_feel", "straight")
        assert result == "straight"


# ---------------------------------------------------------------------------
# _fetch_field_values
# ---------------------------------------------------------------------------


class TestFetchFieldValues:
    def test_fetches_tracks_fields(self, db_with_full_track):
        criterion = _make_groove_criterion()
        values = _fetch_field_values(criterion, TRACK_ID, db_with_full_track)
        assert values["tracks.self_similarity_score"] == pytest.approx(0.65)
        assert values["tracks.beat_regularity"] == pytest.approx(0.84)
        assert values["tracks.danceability"] == pytest.approx(0.72)
        assert values["tracks.groove_consistency"] == pytest.approx(0.74)

    def test_returns_none_for_null_field(self, db_with_full_track):
        criterion = Criterion(
            id="test",
            name="test",
            description="test",
            weight=1.0,
            rule="llm",
            db_fields=["tracks.loudness_db"],
            prompt_hint="test",
        )
        values = _fetch_field_values(criterion, TRACK_ID, db_with_full_track)
        # loudness_db was not inserted
        assert values["tracks.loudness_db"] is None

    def test_fetches_sections_label_sequence(self, db_with_full_track):
        criterion = Criterion(
            id="structure",
            name="Structure",
            description="test",
            weight=1.5,
            rule="llm",
            db_fields=["sections.label"],
            prompt_hint="Evaluate structure.",
        )
        values = _fetch_field_values(criterion, TRACK_ID, db_with_full_track)
        assert values["sections.label"] is not None
        assert "intro" in values["sections.label"]
        assert "chorus" in values["sections.label"]

    def test_fetches_beat_patterns(self, db_with_full_track):
        criterion = Criterion(
            id="rhythm",
            name="Rhythm",
            description="test",
            weight=1.0,
            rule="llm",
            db_fields=["beat_patterns.kick_pattern", "beat_patterns.syncopation_score"],
            prompt_hint="Evaluate rhythm.",
        )
        values = _fetch_field_values(criterion, TRACK_ID, db_with_full_track)
        assert values["beat_patterns.kick_pattern"] == "1000100010001000"
        assert values["beat_patterns.syncopation_score"] == pytest.approx(0.25)

    def test_fetches_chord_progressions(self, db_with_full_track):
        criterion = Criterion(
            id="harmony",
            name="Harmony",
            description="test",
            weight=0.75,
            rule="llm",
            db_fields=["chord_progressions.progression"],
            prompt_hint="Evaluate harmony.",
        )
        values = _fetch_field_values(criterion, TRACK_ID, db_with_full_track)
        assert values["chord_progressions.progression"] is not None
        assert "C - G - Am - F" in values["chord_progressions.progression"]


# ---------------------------------------------------------------------------
# _assemble_user_prompt
# ---------------------------------------------------------------------------


class TestAssembleUserPrompt:
    def test_contains_criterion_name(self):
        criterion = _make_groove_criterion()
        values = {
            "tracks.self_similarity_score": 0.65,
            "tracks.beat_regularity": 0.84,
            "tracks.danceability": 0.72,
            "tracks.groove_consistency": 0.74,
        }
        prompt = _assemble_user_prompt(criterion, values, [])
        assert "Continuous Dance Groove" in prompt

    def test_contains_field_values(self):
        criterion = _make_groove_criterion()
        values = {
            "tracks.self_similarity_score": 0.65,
            "tracks.beat_regularity": 0.84,
            "tracks.danceability": 0.72,
            "tracks.groove_consistency": 0.74,
        }
        prompt = _assemble_user_prompt(criterion, values, [])
        assert "0.650" in prompt
        assert "0.840" in prompt

    def test_contains_prompt_hint(self):
        criterion = _make_groove_criterion()
        values = {"tracks.danceability": 0.72}
        prompt = _assemble_user_prompt(criterion, values, [])
        assert "Evaluate the groove consistency" in prompt

    def test_null_field_note_included_when_present(self):
        criterion = _make_groove_criterion()
        values = {
            "tracks.self_similarity_score": None,
            "tracks.beat_regularity": 0.84,
            "tracks.danceability": 0.72,
            "tracks.groove_consistency": 0.74,
        }
        null_fields = ["tracks.self_similarity_score"]
        prompt = _assemble_user_prompt(criterion, values, null_fields)
        assert "not available" in prompt
        assert "NOTE" in prompt

    def test_no_null_note_when_all_present(self):
        criterion = _make_groove_criterion()
        values = {
            "tracks.self_similarity_score": 0.65,
            "tracks.beat_regularity": 0.84,
            "tracks.danceability": 0.72,
            "tracks.groove_consistency": 0.74,
        }
        prompt = _assemble_user_prompt(criterion, values, [])
        assert "NOTE" not in prompt

    def test_contains_criterion_description(self):
        criterion = _make_groove_criterion()
        values = {"tracks.danceability": 0.5}
        prompt = _assemble_user_prompt(criterion, values, [])
        assert "groove must run all the way through" in prompt


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_returns_prompt_package(self, db_with_full_track):
        criterion = _make_groove_criterion()
        mode = _make_mode_config()
        result = build_prompt(criterion, mode, TRACK_ID, db_with_full_track)
        assert isinstance(result, PromptPackage)
        assert result.criterion_id == "groove"

    def test_system_prompt_from_mode(self, db_with_full_track):
        criterion = _make_groove_criterion()
        mode = _make_mode_config()
        result = build_prompt(criterion, mode, TRACK_ID, db_with_full_track)
        assert result.system_prompt == "You are a test scorer. Respond with JSON only."

    def test_user_prompt_contains_field_values(self, db_with_full_track):
        criterion = _make_groove_criterion()
        mode = _make_mode_config()
        result = build_prompt(criterion, mode, TRACK_ID, db_with_full_track)
        assert "0.650" in result.user_prompt  # self_similarity_score
        assert "0.840" in result.user_prompt  # beat_regularity

    def test_has_null_fields_flag(self, db_with_full_track):
        # loudness_db is not in DB, so null
        criterion = Criterion(
            id="test",
            name="test",
            description="test",
            weight=1.0,
            rule="llm",
            db_fields=["tracks.loudness_db", "tracks.bpm"],
            prompt_hint="test",
        )
        mode = ModeConfig(name="test", description="test", system_prompt="test", criteria=[criterion])
        result = build_prompt(criterion, mode, TRACK_ID, db_with_full_track)
        assert result.has_null_fields is True
        assert "tracks.loudness_db" in result.null_field_names

    def test_raises_for_non_llm_criterion(self, db_with_full_track):
        criterion = Criterion(
            id="bpm", name="BPM", description="test", weight=1.5, rule="lte", db_field="tracks.bpm", threshold=135
        )
        mode = _make_mode_config()
        with pytest.raises(ValueError, match="non-llm criterion"):
            build_prompt(criterion, mode, TRACK_ID, db_with_full_track)


# ---------------------------------------------------------------------------
# build_all_llm_prompts
# ---------------------------------------------------------------------------


class TestBuildAllLlmPrompts:
    def test_returns_one_package_per_llm_criterion(self, db_with_full_track):
        mode = ModeConfig(
            name="test",
            description="test",
            system_prompt="test",
            criteria=[
                _make_groove_criterion(),
                Criterion(
                    id="bpm",
                    name="BPM",
                    description="test",
                    weight=1.5,
                    rule="lte",
                    db_field="tracks.bpm",
                    threshold=135,
                ),
                Criterion(
                    id="structure",
                    name="Structure",
                    description="test",
                    weight=1.5,
                    rule="llm",
                    db_fields=["sections.label"],
                    prompt_hint="Evaluate structure.",
                ),
            ],
        )
        packages = build_all_llm_prompts(mode, TRACK_ID, db_with_full_track)
        # Only llm criteria get packages — bpm (lte) is excluded
        assert len(packages) == 2
        ids = {p.criterion_id for p in packages}
        assert "groove" in ids
        assert "structure" in ids
        assert "bpm" not in ids

    def test_handles_failure_gracefully(self, tmp_path):
        """If one criterion fails, others should still be returned."""
        db_path = tmp_path / "empty.db"
        conn = get_connection(db_path)
        conn.close()

        mode = ModeConfig(
            name="test",
            description="test",
            system_prompt="test",
            criteria=[
                _make_groove_criterion(),
            ],
        )
        # Should not raise even with missing track data
        packages = build_all_llm_prompts(mode, "nonexistent_track_id", db_path)
        # Returns packages (possibly with null fields) rather than raising
        assert isinstance(packages, list)
