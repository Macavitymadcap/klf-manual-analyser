"""
analysis/energy.py — Stage 3a: energy analysis.

Responsibilities:
  - Compute RMS energy profile (sampled every 0.5s, normalised 0.0–1.0)
  - Measure integrated loudness in LUFS using pyloudnorm
  - Compute dynamic range
  - Estimate verse/chorus energy delta (requires sections from harmony.py,
    falls back to first/second half comparison if sections not yet labelled)
  - Classify energy shape ("building", "flat", "peaked", "unclear")
  - Write scalar results to tracks table
  - Write RMS profile JSON blob to tracks_timeseries table

Writes to SQLite:
  UPDATE tracks SET
    loudness_db, dynamic_range_db, verse_chorus_delta, energy_shape
  WHERE track_id = ?

  INSERT OR REPLACE INTO tracks_timeseries (track_id, rms_profile_json)
  VALUES (?, ?)

Error handling (per docs/ERROR_HANDLING.md):
  - Numerical errors → write null for affected fields, log warning, continue
  - Unhandled exception → write null for all fields, log error, continue
"""

import logging
from pathlib import Path

from manual_analyser.analysis.energy.features import (
    _classify_energy_shape,
    _compute_dynamic_range,
    _compute_energy,
    _delta_from_halves,
)
from manual_analyser.analysis.energy.types import EnergyResult
from manual_analyser.analysis.energy.writer import _write_nulls, _write_result

logger = logging.getLogger(__name__)


def analyse_energy(
    track_id: str,
    full_wav: Path,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
) -> EnergyResult | None:
    """
    Compute energy features from the full mix WAV and write to SQLite.

    The verse/chorus delta is computed from section labels if they exist
    in the database (written by harmony.py). If no sections exist yet,
    falls back to comparing the first and second halves of the track.
    This means energy.py can run in any order relative to harmony.py
    without error — the delta estimate just becomes less accurate.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        full_wav: Path to the decoded mono WAV.
        data_dir: Root data directory (default: "data/").
        db_path: Path to SQLite database. Defaults to data/manual_analyser.db.

    Returns:
        EnergyResult on success, or None if analysis failed.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        result = _compute_energy(full_wav, short_id, resolved_db, track_id)
    except Exception as e:
        logger.exception("[%s] [energy] Analysis failed: %s", short_id, e, exc_info=True)
        _write_nulls(resolved_db, track_id, short_id)
        return None

    _write_result(resolved_db, track_id, result, short_id)
    logger.info(
        "[%s] [energy] loudness=%.2f range=%.2f delta=%.2f shape=%s profile_len=%d",
        short_id,
        result.loudness_db,
        result.dynamic_range_db,
        result.verse_chorus_delta,
        result.energy_shape,
        len(result.rms_profile),
    )
    return result


__all__ = [
    "_classify_energy_shape",
    "_compute_dynamic_range",
    "_delta_from_halves",
]
