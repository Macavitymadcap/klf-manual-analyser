"""Tests for embedding/embed.py"""

from unittest.mock import patch

import pytest

from manual_analyser.db import get_connection
from manual_analyser.embedding import embed_track, is_qdrant_available
from manual_analyser.embedding.ollama_embed import EmbedUnavailableError
from manual_analyser.embedding.qdrant_client import QdrantUnavailableError
from manual_analyser.embedding.types import EmbedResult, EmbedSkipped

TRACK_ID = "b" * 32
_FAKE_VECTOR = [0.1] * 384
_FAKE_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

_PATCH_VECTOR = "manual_analyser.embedding.ollama_embed.get_vector"
_PATCH_CHECK_QDRANT = "manual_analyser.embedding.qdrant_client.check_qdrant"
_PATCH_CHECK_EMBED = "manual_analyser.embedding.ollama_embed.check_embed_model"
_PATCH_ENSURE = "manual_analyser.embedding.qdrant_client.ensure_collection"
_PATCH_UPSERT = "manual_analyser.embedding.qdrant_client.upsert"


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = get_connection(path)
    with conn:
        conn.execute(
            """INSERT INTO tracks
               (track_id, filename, duration, analysis_timestamp, analysis_version,
                artist, song_name, bpm, key, mode, groove_feel, energy_shape, danceability)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                TRACK_ID,
                "test.mp3",
                180.0,
                "2025-01-01T00:00:00+00:00",
                "0.1.0",
                "The KLF",
                "Doctorin",
                126.0,
                "C",
                "major",
                "straight",
                "building",
                0.75,
            ),
        )
        conn.execute(
            "INSERT INTO sections (track_id, position, start, end, duration, label, label_confidence, label_source)"
            " VALUES (?, 0, 0.0, 30.0, 30.0, 'intro', 0.9, 'hybrid')",
            (TRACK_ID,),
        )
        conn.execute(
            "INSERT INTO beat_patterns (track_id, kick_pattern, snare_pattern, hihat_pattern) VALUES (?, ?, ?, ?)",
            (TRACK_ID, "1000100010001000", "0000100000001000", "1010101010101010"),
        )
    conn.close()
    return path


class TestIsQdrantAvailable:
    def test_returns_true_when_both_available(self):
        with patch(_PATCH_CHECK_QDRANT), patch(_PATCH_CHECK_EMBED):
            assert is_qdrant_available() is True

    def test_returns_false_when_qdrant_down(self):
        with patch(_PATCH_CHECK_QDRANT, side_effect=QdrantUnavailableError("down")):
            assert is_qdrant_available() is False

    def test_returns_false_when_embed_model_missing(self):
        with patch(_PATCH_CHECK_QDRANT), patch(_PATCH_CHECK_EMBED, side_effect=EmbedUnavailableError("missing")):
            assert is_qdrant_available() is False


class TestEmbedTrack:
    def test_success_returns_embed_result(self, db_path):
        with (
            patch(_PATCH_VECTOR, return_value=_FAKE_VECTOR),
            patch(_PATCH_ENSURE),
            patch(_PATCH_UPSERT, return_value=_FAKE_UUID),
        ):
            result = embed_track(TRACK_ID, db_path)
        assert isinstance(result, EmbedResult)
        assert result.track_id == TRACK_ID
        assert result.qdrant_id == _FAKE_UUID

    def test_success_writes_feature_summary(self, db_path):
        with (
            patch(_PATCH_VECTOR, return_value=_FAKE_VECTOR),
            patch(_PATCH_ENSURE),
            patch(_PATCH_UPSERT, return_value=_FAKE_UUID),
        ):
            embed_track(TRACK_ID, db_path)
        conn = get_connection(db_path)
        row = conn.execute("SELECT feature_summary FROM tracks WHERE track_id = ?", (TRACK_ID,)).fetchone()
        conn.close()
        assert row[0] is not None
        assert "The KLF" in row[0]

    def test_success_writes_vector_record(self, db_path):
        with (
            patch(_PATCH_VECTOR, return_value=_FAKE_VECTOR),
            patch(_PATCH_ENSURE),
            patch(_PATCH_UPSERT, return_value=_FAKE_UUID),
        ):
            embed_track(TRACK_ID, db_path)
        conn = get_connection(db_path)
        row = conn.execute("SELECT qdrant_id FROM track_vectors WHERE track_id = ?", (TRACK_ID,)).fetchone()
        conn.close()
        assert row[0] == _FAKE_UUID

    def test_qdrant_unavailable_returns_skipped(self, db_path):
        with (
            patch(_PATCH_VECTOR, return_value=_FAKE_VECTOR),
            patch(_PATCH_ENSURE),
            patch(_PATCH_UPSERT, side_effect=QdrantUnavailableError("down")),
        ):
            result = embed_track(TRACK_ID, db_path)
        assert isinstance(result, EmbedSkipped)
        assert "down" in result.reason

    def test_embed_unavailable_returns_skipped(self, db_path):
        with patch(_PATCH_VECTOR, side_effect=EmbedUnavailableError("no model")):
            result = embed_track(TRACK_ID, db_path)
        assert isinstance(result, EmbedSkipped)

    def test_unexpected_error_returns_skipped(self, db_path):
        with patch(_PATCH_VECTOR, side_effect=RuntimeError("boom")):
            result = embed_track(TRACK_ID, db_path)
        assert isinstance(result, EmbedSkipped)
        assert "unexpected" in result.reason

    def test_missing_track_returns_skipped(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = get_connection(db_path)
        conn.close()
        result = embed_track(TRACK_ID, db_path)
        assert isinstance(result, EmbedSkipped)
