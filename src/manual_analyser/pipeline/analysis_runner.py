"""pipeline/analysis_runner.py — Run all Stage 3 analysis modules for one track."""

import logging
from pathlib import Path

from manual_analyser.analysis import energy, groove, harmony, rhythm, structure, tempo
from manual_analyser.audio.separate import SeparateResult
from manual_analyser.pipeline.types import TrackState

logger = logging.getLogger(__name__)

_MODULES = [
    ("tempo", lambda stems, db: tempo.analyse_tempo(stems.track_id, stems.stem_dir / "full.wav", db_path=db)),
    ("rhythm", lambda stems, db: rhythm.analyse_rhythm(stems.track_id, stems.stem_dir / "drums.wav", db_path=db)),
    ("energy", lambda stems, db: energy.analyse_energy(stems.track_id, stems.stem_dir / "full.wav", db_path=db)),
    ("harmony", lambda stems, db: harmony.analyse_harmony(stems.track_id, stems.stem_dir / "full.wav", db_path=db)),
    ("groove", lambda stems, db: groove.analyse_groove(stems.track_id, stems.stem_dir / "full.wav", db_path=db)),
    ("structure", lambda stems, db: structure.segment_track(stems.track_id, stems.stem_dir / "full.wav", db_path=db)),
]


def run_analysis(stems: SeparateResult, db_path: Path, state: TrackState) -> None:
    """
    Run all Stage 3a analysis modules for one track.

    Each module failure is caught and recorded — they are independent.
    Updates state.failed_stages and state.status in place.
    """
    for name, fn in _MODULES:
        _run_module(name, fn, stems, db_path, state)


def _run_module(name: str, fn, stems: SeparateResult, db_path: Path, state: TrackState) -> None:
    try:
        fn(stems, db_path)
        logger.info("[%s] [%s] done", state.short_id, name)
    except Exception as exc:
        logger.error("[%s] [%s] failed: %s", state.short_id, name, exc, exc_info=True)
        state.failed_stages.append(name)
