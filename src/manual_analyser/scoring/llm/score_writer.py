"""scoring/llm/score_writer.py — Write LLM scoring results to SQLite."""

import logging
from pathlib import Path

from manual_analyser.audio.decode import utc_now_iso
from manual_analyser.db import get_connection
from manual_analyser.scoring.llm.types import LlmFailure, LlmResult, LlmScore

logger = logging.getLogger(__name__)


def write_result(track_id: str, mode: str, result: LlmResult, db_path: Path) -> None:
    """Write a single LlmScore or LlmFailure to the scores table."""
    if isinstance(result, LlmScore):
        _write_score(track_id, mode, result, db_path)
    else:
        _write_failure(track_id, mode, result, db_path)


def _write_score(track_id: str, mode: str, score: LlmScore, db_path: Path) -> None:
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scores
                    (track_id, mode, criterion_id, score, reasoning, passed, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    track_id,
                    mode,
                    score.criterion_id,
                    round(score.score, 4),
                    score.reasoning,
                    int(score.passed),
                    utc_now_iso(),
                ),
            )
    finally:
        conn.close()


def _write_failure(track_id: str, mode: str, failure: LlmFailure, db_path: Path) -> None:
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scores
                    (track_id, mode, criterion_id, score, reasoning, passed, scored_at)
                VALUES (?, ?, ?, NULL, ?, 0, ?)
                """,
                (track_id, mode, failure.criterion_id, failure.reason, utc_now_iso()),
            )
    finally:
        conn.close()
