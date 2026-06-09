"""Tests for pipeline/cache.py"""

import pytest

from manual_analyser.db import get_connection
from manual_analyser.pipeline.cache import (
    scores_exist,
    sections_labelled,
    track_in_db,
    transcript_in_db,
    vector_in_qdrant,
)
from manual_analyser.utils import utc_now_iso

TRACK_ID = "a" * 32


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = get_connection(path)
    with conn:
        conn.execute(
            "INSERT INTO tracks (track_id, filename, duration, analysis_timestamp, analysis_version)"
            " VALUES (?, ?, ?, ?, ?)",
            (TRACK_ID, "test.mp3", 180.0, utc_now_iso(), "0.1.0"),
        )
    conn.close()
    return path


@pytest.fixture
def empty_db(tmp_path):
    path = tmp_path / "empty.db"
    conn = get_connection(path)
    conn.close()
    return path


class TestTrackInDb:
    def test_true_when_present(self, db_path):
        assert track_in_db(TRACK_ID, db_path) is True

    def test_false_when_absent(self, empty_db):
        assert track_in_db(TRACK_ID, empty_db) is False


class TestTranscriptInDb:
    def test_false_when_no_segments(self, db_path):
        assert transcript_in_db(TRACK_ID, db_path) is False

    def test_true_when_segments_exist(self, db_path):
        conn = get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO transcript_segments (track_id, start, end, text) VALUES (?, 0.0, 2.0, 'hello')",
                (TRACK_ID,),
            )
        conn.close()
        assert transcript_in_db(TRACK_ID, db_path) is True


class TestSectionsLabelled:
    def test_false_when_no_sections(self, db_path):
        assert sections_labelled(TRACK_ID, db_path) is False

    def test_false_when_all_unknown(self, db_path):
        conn = get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO sections (track_id, position, start, end, duration, label, label_confidence, label_source)"
                " VALUES (?, 0, 0.0, 30.0, 30.0, 'unknown', 0.0, 'acoustic')",
                (TRACK_ID,),
            )
        conn.close()
        assert sections_labelled(TRACK_ID, db_path) is False

    def test_true_when_labelled_section_exists(self, db_path):
        conn = get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO sections (track_id, position, start, end, duration, label, label_confidence, label_source)"
                " VALUES (?, 0, 0.0, 30.0, 30.0, 'chorus', 0.9, 'hybrid')",
                (TRACK_ID,),
            )
        conn.close()
        assert sections_labelled(TRACK_ID, db_path) is True


class TestVectorInQdrant:
    def test_false_when_no_record(self, db_path):
        assert vector_in_qdrant(TRACK_ID, db_path) is False

    def test_true_when_record_exists(self, db_path):
        conn = get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO track_vectors (track_id, qdrant_id) VALUES (?, ?)",
                (TRACK_ID, "some-uuid"),
            )
        conn.close()
        assert vector_in_qdrant(TRACK_ID, db_path) is True


class TestScoresExist:
    def test_false_when_no_scores(self, db_path):
        assert scores_exist(TRACK_ID, "1988", db_path) is False

    def test_true_when_scores_present(self, db_path):
        conn = get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO scores (track_id, mode, criterion_id, score, passed, scored_at)"
                " VALUES (?, '1988', 'bpm', 1.0, 1, ?)",
                (TRACK_ID, utc_now_iso()),
            )
        conn.close()
        assert scores_exist(TRACK_ID, "1988", db_path) is True

    def test_false_for_different_mode(self, db_path):
        conn = get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO scores (track_id, mode, criterion_id, score, passed, scored_at)"
                " VALUES (?, '1988', 'bpm', 1.0, 1, ?)",
                (TRACK_ID, utc_now_iso()),
            )
        conn.close()
        assert scores_exist(TRACK_ID, "contemporary", db_path) is False
