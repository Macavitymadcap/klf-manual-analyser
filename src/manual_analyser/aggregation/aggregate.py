"""
aggregation/aggregate.py — Stage 7: aggregate scores and generate recipe.

Public API:
    aggregate(mode, db_path, use_qdrant) -> AggregateReport

Raises:
    InsufficientDataError — fewer than 2 scored tracks; caller still renders
                            per-track report without aggregate view.
"""

import logging
from pathlib import Path

from manual_analyser.aggregation import clusters as cluster_module
from manual_analyser.aggregation import recipe as recipe_module
from manual_analyser.aggregation.queries import (
    fetch_criterion_ids,
    fetch_criterion_stats,
    fetch_modal_structure,
    fetch_modal_track_features,
    fetch_scored_track_ids,
    fetch_track_metadata,
    fetch_track_score_summary,
)
from manual_analyser.aggregation.types import (
    AggregateReport,
    ClusterInfo,
    CriterionSummary,
    TrackSummary,
)

logger = logging.getLogger(__name__)

__all__ = ["aggregate", "InsufficientDataError"]

MIN_TRACKS = 2


class InsufficientDataError(Exception):
    """Raised when fewer than MIN_TRACKS have been scored."""


def aggregate(mode: str, db_path: Path, use_qdrant: bool = False) -> AggregateReport:
    """
    Build a complete AggregateReport from scored tracks in the DB.

    Raises InsufficientDataError if fewer than 2 tracks are scored.
    Never raises for recipe or cluster failures — those are captured in the report.
    """
    track_ids = fetch_scored_track_ids(mode, db_path)
    if len(track_ids) < MIN_TRACKS:
        raise InsufficientDataError(
            f"Need at least {MIN_TRACKS} scored tracks for aggregation; found {len(track_ids)}."
        )

    logger.info("[aggregation] Aggregating %d tracks for mode '%s'", len(track_ids), mode)

    criteria = _build_criteria_summaries(mode, db_path)
    tracks = _build_track_summaries(track_ids, mode, db_path)
    modal = fetch_modal_track_features(db_path)
    structure = fetch_modal_structure(db_path)

    report = AggregateReport(
        mode=mode,
        track_count=len(track_ids),
        criteria=criteria,
        tracks=tracks,
        modal_bpm=modal.get("modal_bpm"),
        modal_key=modal.get("modal_key"),
        modal_mode=modal.get("modal_mode"),
        modal_groove_feel=modal.get("modal_groove_feel"),
        modal_energy_shape=modal.get("modal_energy_shape"),
        modal_structure=structure,
        recipe=None,
        recipe_error=None,
    )

    _add_recipe(report)

    if use_qdrant:
        report.clusters = _fetch_clusters(track_ids)

    return report


def _build_criteria_summaries(mode: str, db_path: Path) -> list[CriterionSummary]:
    criterion_ids = fetch_criterion_ids(mode, db_path)
    summaries = []
    for cid in criterion_ids:
        stats = fetch_criterion_stats(cid, mode, db_path)
        summaries.append(
            CriterionSummary(
                criterion_id=cid,
                pass_rate=stats["pass_rate"],
                mean_score=stats["mean_score"],
                scored_track_count=stats["scored_track_count"],
            )
        )
    return summaries


def _build_track_summaries(track_ids: list[str], mode: str, db_path: Path) -> list[TrackSummary]:
    summaries = []
    for tid in track_ids:
        meta = fetch_track_metadata(tid, db_path)
        scores = fetch_track_score_summary(tid, mode, db_path)
        summaries.append(
            TrackSummary(
                track_id=tid,
                artist=meta.get("artist"),
                song_name=meta.get("song_name"),
                overall_score=scores["overall_score"],
                passed_count=scores["passed_count"],
                total_count=scores["total_count"],
            )
        )
    return sorted(summaries, key=lambda t: t.overall_score, reverse=True)


def _add_recipe(report: AggregateReport) -> None:
    try:
        report.recipe = recipe_module.generate_recipe(report)
        logger.info("[aggregation] Recipe generated (%d chars)", len(report.recipe))
    except Exception as exc:
        logger.error("[aggregation] Recipe generation failed: %s", exc)
        report.recipe_error = str(exc)


def _fetch_clusters(track_ids: list[str]) -> list[ClusterInfo]:
    clusters = cluster_module.fetch_clusters(track_ids)
    logger.info("[aggregation] %d clusters found", len(clusters))
    return clusters
