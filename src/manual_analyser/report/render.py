"""
report/render.py — Stage 8: render HTML report from SQLite data.

Public API:
    render(mode, db_path, data_dir) -> Path   (path to summary HTML)

Writes:
    data/reports/{mode}_{timestamp}/index.html
    data/reports/{mode}_{timestamp}/track_{track_id}.html  (one per track)
"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from manual_analyser.aggregation.aggregate import InsufficientDataError, aggregate
from manual_analyser.audio.decode import utc_now_iso
from manual_analyser.report.queries import (
    load_criterion_summaries,
    load_scored_track_ids,
    load_track_report_data,
    load_track_row,
)
from manual_analyser.report.types import SummaryReportData

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

__all__ = ["render", "RenderError"]


class RenderError(Exception):
    """Raised when report rendering fails hard (template error, unwritable dir)."""


def render(mode: str, db_path: Path, data_dir: Path) -> Path:
    """
    Render the full HTML report for a mode.

    Returns the path to the generated index.html.
    Raises RenderError on hard failures.
    """
    out_dir = _make_output_dir(data_dir, mode)
    env = _make_env()

    try:
        agg = aggregate(mode, db_path, use_qdrant=False)
    except InsufficientDataError as exc:
        logger.warning("[render] %s — rendering per-track report only", exc)
        agg = None

    track_ids = load_scored_track_ids(mode, db_path)
    logger.info("[render] Rendering %d track pages for mode '%s'", len(track_ids), mode)

    for track_id in track_ids:
        _render_track(track_id, mode, db_path, out_dir, env)

    summary_path = _render_summary(agg, track_ids, mode, db_path, out_dir, env)
    logger.info("[render] Report written to %s", summary_path)
    return summary_path


def _make_output_dir(data_dir: Path, mode: str) -> Path:
    ts = utc_now_iso().replace(":", "-").replace("+", "").split(".")[0]
    out_dir = data_dir / "reports" / f"{mode}_{ts}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RenderError(f"Cannot create report directory {out_dir}: {exc}") from exc
    return out_dir


def _make_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _render_track(
    track_id: str,
    mode: str,
    db_path: Path,
    out_dir: Path,
    env: Environment,
) -> None:
    stem_base_url = f"/stems/{track_id}"
    try:
        data = load_track_report_data(track_id, mode, db_path, stem_base_url)
        tmpl = env.get_template("track.html")
        html = tmpl.render(**vars(data))
        out_path = out_dir / f"track_{track_id}.html"
        out_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        logger.exception("[render] Failed to render track %s: %s", track_id[:8], exc, exc_info=True)


def _render_summary(agg, track_ids, mode, db_path, out_dir, env) -> Path:
    criteria = load_criterion_summaries(mode, db_path)

    track_rows = []
    for tid in track_ids:
        detail_url = f"track_{tid}.html"
        row = load_track_row(tid, mode, db_path, detail_url)
        track_rows.append(row)

    overall_pass_rate = 0.0
    if criteria:
        overall_pass_rate = sum(c.pass_rate for c in criteria) / len(criteria)

    data = SummaryReportData(
        mode_name=mode,
        track_count=len(track_ids),
        rendered_at=utc_now_iso(),
        recipe=agg.recipe if agg else None,
        recipe_error=agg.recipe_error if agg else None,
        modal_bpm=agg.modal_bpm if agg else None,
        modal_key=agg.modal_key if agg else None,
        modal_mode=agg.modal_mode if agg else None,
        modal_groove_feel=agg.modal_groove_feel if agg else None,
        modal_energy_shape=agg.modal_energy_shape if agg else None,
        modal_structure=agg.modal_structure if agg else [],
        overall_pass_rate=round(overall_pass_rate, 3),
        criteria=criteria,
        tracks=track_rows,
    )

    try:
        tmpl = env.get_template("summary.html")
        html = tmpl.render(**vars(data))
    except Exception as exc:
        raise RenderError(f"Template error in summary.html: {exc}") from exc

    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
