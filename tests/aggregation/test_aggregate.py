"""Tests for aggregation/aggregate.py"""

from unittest.mock import patch

import pytest

from manual_analyser.aggregation.aggregate import InsufficientDataError, aggregate
from manual_analyser.aggregation.types import AggregateReport
from manual_analyser.db import get_connection
from manual_analyser.utils import utc_now_iso

_PATCH_RECIPE = "manual_analyser.aggregation.recipe.generate_recipe"
_PATCH_CLUSTERS = "manual_analyser.aggregation.clusters.fetch_clusters"

TRACKS = [
    ("a" * 32, "The KLF", "Doctorin The Tardis", 126.0, "C", "major", "straight"),
    ("b" * 32, "KLF", "3AM Eternal", 130.0, "G", "major", "straight"),
    ("c" * 32, "KLF", "What Time Is Love", 128.0, "C", "minor", "straight"),
]


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = get_connection(path)
    with conn:
        for tid, artist, song, bpm, key, mode, groove in TRACKS:
            conn.execute(
                """INSERT INTO tracks
                   (track_id, filename, duration, analysis_timestamp, analysis_version,
                    artist, song_name, bpm, key, mode, groove_feel, energy_shape)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tid, f"{artist}.mp3", 180.0, utc_now_iso(), "0.1.0", artist, song, bpm, key, mode, groove, "building"),
            )
        _insert_scores(conn)
    conn.close()
    return path


def _insert_scores(conn) -> None:
    for tid, *_ in TRACKS:
        for cid, score, passed in [("bpm", 1.0, 1), ("groove", 0.7, 1), ("structure", 0.4, 0)]:
            conn.execute(
                "INSERT INTO scores (track_id, mode, criterion_id, score, passed, scored_at)"
                " VALUES (?, '1988', ?, ?, ?, ?)",
                (tid, cid, score, passed, utc_now_iso()),
            )


class TestAggregate:
    def test_returns_aggregate_report(self, db_path):
        with patch(_PATCH_RECIPE, return_value="DO THIS."):
            result = aggregate("1988", db_path)
        assert isinstance(result, AggregateReport)

    def test_track_count_correct(self, db_path):
        with patch(_PATCH_RECIPE, return_value="DO THIS."):
            result = aggregate("1988", db_path)
        assert result.track_count == 3

    def test_criteria_summaries_populated(self, db_path):
        with patch(_PATCH_RECIPE, return_value="DO THIS."):
            result = aggregate("1988", db_path)
        criterion_ids = {c.criterion_id for c in result.criteria}
        assert "bpm" in criterion_ids
        assert "groove" in criterion_ids

    def test_bpm_pass_rate_correct(self, db_path):
        with patch(_PATCH_RECIPE, return_value="DO THIS."):
            result = aggregate("1988", db_path)
        bpm = next(c for c in result.criteria if c.criterion_id == "bpm")
        assert bpm.pass_rate == pytest.approx(1.0)

    def test_structure_pass_rate_correct(self, db_path):
        with patch(_PATCH_RECIPE, return_value="DO THIS."):
            result = aggregate("1988", db_path)
        structure = next(c for c in result.criteria if c.criterion_id == "structure")
        assert structure.pass_rate == pytest.approx(0.0)

    def test_tracks_sorted_by_score_descending(self, db_path):
        with patch(_PATCH_RECIPE, return_value="DO THIS."):
            result = aggregate("1988", db_path)
        scores = [t.overall_score for t in result.tracks]
        assert scores == sorted(scores, reverse=True)

    def test_recipe_populated_on_success(self, db_path):
        with patch(_PATCH_RECIPE, return_value="KEEP IT SIMPLE."):
            result = aggregate("1988", db_path)
        assert result.recipe == "KEEP IT SIMPLE."
        assert result.recipe_error is None

    def test_recipe_error_captured_on_failure(self, db_path):
        with patch(_PATCH_RECIPE, side_effect=Exception("Ollama down")):
            result = aggregate("1988", db_path)
        assert result.recipe is None
        assert "Ollama down" in result.recipe_error

    def test_modal_key_populated(self, db_path):
        with patch(_PATCH_RECIPE, return_value="x"):
            result = aggregate("1988", db_path)
        assert result.modal_key == "C"

    def test_modal_groove_populated(self, db_path):
        with patch(_PATCH_RECIPE, return_value="x"):
            result = aggregate("1988", db_path)
        assert result.modal_groove_feel == "straight"

    def test_clusters_empty_when_qdrant_not_requested(self, db_path):
        with patch(_PATCH_RECIPE, return_value="x"):
            result = aggregate("1988", db_path, use_qdrant=False)
        assert result.clusters == []

    def test_clusters_populated_when_qdrant_requested(self, db_path):
        from manual_analyser.aggregation.types import ClusterInfo

        fake_clusters = [ClusterInfo(0, ["a" * 32], {"groove_feel": "straight"})]
        with patch(_PATCH_RECIPE, return_value="x"), patch(_PATCH_CLUSTERS, return_value=fake_clusters):
            result = aggregate("1988", db_path, use_qdrant=True)
        assert len(result.clusters) == 1


class TestInsufficientData:
    def test_raises_with_zero_tracks(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = get_connection(db_path)
        conn.close()
        with pytest.raises(InsufficientDataError):
            aggregate("1988", db_path)

    def test_raises_with_one_track(self, tmp_path):
        db_path = tmp_path / "one.db"
        conn = get_connection(db_path)
        tid = "a" * 32
        with conn:
            conn.execute(
                "INSERT INTO tracks (track_id, filename, duration, analysis_timestamp, analysis_version)"
                " VALUES (?, ?, ?, ?, ?)",
                (tid, "test.mp3", 180.0, utc_now_iso(), "0.1.0"),
            )
            conn.execute(
                "INSERT INTO scores (track_id, mode, criterion_id, score, passed, scored_at)"
                " VALUES (?, '1988', 'bpm', 1.0, 1, ?)",
                (tid, utc_now_iso()),
            )
        conn.close()
        with pytest.raises(InsufficientDataError):
            aggregate("1988", db_path)
