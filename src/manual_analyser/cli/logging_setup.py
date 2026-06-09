"""cli/logging_setup.py — Configure logging for a pipeline run."""

import logging
from pathlib import Path


def configure(data_dir: Path, verbose: bool = False) -> None:
    """Set up file + stderr logging for a pipeline run."""
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "pipeline.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("manual_analyser").setLevel(level)
