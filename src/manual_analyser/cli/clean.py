"""cli/clean.py — Implementation of the clean subcommand."""

import shutil
from pathlib import Path

from manual_analyser.cli.output import console, print_info
from manual_analyser.db import get_connection


def clean(
    data_dir: Path,
    stems: bool,
    features: bool,
    reports: bool,
) -> None:
    """
    Remove cached data. If no flags set, removes everything.

    Args:
        data_dir: Root data directory.
        stems: Remove WAV stems from data/stems/.
        features: Remove all track records from the DB.
        reports: Remove rendered HTML from data/reports/.
    """
    remove_all = not any([stems, features, reports])

    if stems or remove_all:
        _remove_stems(data_dir)

    if features or remove_all:
        _remove_features(data_dir)

    if reports or remove_all:
        _remove_reports(data_dir)


def _remove_stems(data_dir: Path) -> None:
    stems_dir = data_dir / "stems"
    if stems_dir.exists():
        shutil.rmtree(stems_dir)
        print_info(f"Removed {stems_dir}")
    else:
        console.print("  No stems directory found.")


def _remove_features(data_dir: Path) -> None:
    db_path = data_dir / "manual_analyser.db"
    if not db_path.exists():
        console.print("  No database found.")
        return
    conn = get_connection(db_path)
    try:
        with conn:
            for table in (
                "scores",
                "transcript_segments",
                "chord_progressions",
                "sections",
                "beat_patterns",
                "tracks_timeseries",
                "track_vectors",
                "tracks",
            ):
                conn.execute(f"DELETE FROM {table}")
        print_info(f"Cleared all track records from {db_path}")
    finally:
        conn.close()


def _remove_reports(data_dir: Path) -> None:
    reports_dir = data_dir / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
        print_info(f"Removed {reports_dir}")
    else:
        console.print("  No reports directory found.")
