import json
import logging
from pathlib import Path

from manual_analyser.analysis.energy.types import EnergyResult
from manual_analyser.db.connection import get_connection

logger = logging.getLogger(__name__)


def _write_result(
    db_path: Path,
    track_id: str,
    result: EnergyResult,
    short_id: str,
) -> None:
    """Write EnergyResult to tracks and tracks_timeseries tables."""
    rms_json = json.dumps([round(v, 4) for v in result.rms_profile])

    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET
                    loudness_db = ?,
                    dynamic_range_db = ?,
                    verse_chorus_delta = ?,
                    energy_shape = ?
                WHERE track_id = ?
                """,
                (
                    round(result.loudness_db, 4),
                    round(result.dynamic_range_db, 4),
                    round(result.verse_chorus_delta, 4),
                    result.energy_shape,
                    track_id,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO tracks_timeseries (track_id, rms_profile_json)
                VALUES (?, ?)
                """,
                (track_id, rms_json),
            )
    finally:
        conn.close()


def _write_nulls(db_path: Path, track_id: str, short_id: str) -> None:
    """Write null for all energy fields when analysis fails."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET
                    loudness_db = NULL,
                    dynamic_range_db = NULL,
                    verse_chorus_delta = NULL,
                    energy_shape = NULL
                WHERE track_id = ?
                """,
                (track_id,),
            )
        logger.warning("[%s] [energy] Wrote null fields due to analysis failure", short_id)
    finally:
        conn.close()
