"""
pipeline/__init__.py — Stage orchestrator for the KLF Manual Analyser.

Public API:
    run_pipeline(mp3_paths, mode, db_path, data_dir, no_cache, use_qdrant) -> RunSummary

Hard aborts (re-raised to CLI):
    DecodeAbortError  — ffmpeg missing
    SeparateAbortError — Demucs model missing
    OllamaUnavailableError — Ollama not running or model not pulled

Per-track isolation: all other failures are caught, recorded in TrackState,
and the pipeline continues with the next track.
"""

import logging
from pathlib import Path

from manual_analyser.analysis.structure import align_sections
from manual_analyser.audio.decode import DecodeAbortError, DecodeSkipError, decode_track
from manual_analyser.audio.separate import SeparateAbortError, SeparateSkipError, separate_track
from manual_analyser.embedding import embed_track, is_qdrant_available
from manual_analyser.pipeline import analysis_runner, cache, scoring_runner
from manual_analyser.pipeline.types import RunSummary, TrackState, TrackStatus
from manual_analyser.transcription.whisper import transcribe_track

logger = logging.getLogger(__name__)

__all__ = ["run_pipeline", "RunSummary", "TrackStatus"]


def run_pipeline(
    mp3_paths: list[Path],
    mode: str,
    db_path: Path,
    data_dir: Path,
    no_cache: bool = False,
    use_qdrant: bool | None = None,
) -> RunSummary:
    """
    Run the full analysis pipeline for a list of MP3 paths.

    Returns a RunSummary with complete/partial/skipped track lists.
    Raises DecodeAbortError or SeparateAbortError on hard failures.
    """
    if use_qdrant is None:
        use_qdrant = is_qdrant_available()

    states = [TrackState(track_id="", mp3_path=str(p)) for p in mp3_paths]

    for i, mp3_path in enumerate(mp3_paths):
        states[i] = _process_track(mp3_path, mode, db_path, data_dir, no_cache, use_qdrant)

    return _build_summary(states)


def _process_track(
    mp3_path: Path,
    mode: str,
    db_path: Path,
    data_dir: Path,
    no_cache: bool,
    use_qdrant: bool,
) -> TrackState:
    """Run all pipeline stages for one track. Returns final TrackState."""
    state = TrackState(track_id="", mp3_path=str(mp3_path))

    decode_result = _stage_decode(mp3_path, db_path, data_dir, no_cache, state)
    if decode_result is None:
        return state

    state.track_id = decode_result.track_id
    state.status = TrackStatus.SEPARATING

    stems = _stage_separate(decode_result, data_dir, no_cache, state)
    if stems is None:
        return state

    _stage_analyse(stems, db_path, no_cache, state)
    _stage_transcribe(stems, db_path, no_cache, state)
    _stage_align(state.track_id, db_path, no_cache, state)
    _stage_embed(state.track_id, db_path, use_qdrant, no_cache, state)
    _stage_score(state.track_id, mode, db_path, no_cache, state)

    state.status = TrackStatus.PARTIAL if state.failed_stages else TrackStatus.COMPLETE
    return state


def _stage_decode(mp3_path, db_path, data_dir, no_cache, state):
    state.status = TrackStatus.DECODING
    try:
        return decode_track(mp3_path, data_dir=data_dir, db_path=db_path, no_cache=no_cache)
    except DecodeAbortError:
        raise
    except DecodeSkipError as exc:
        logger.warning("[pipeline] Skipping %s: %s", mp3_path.name, exc)
        state.status = TrackStatus.SKIPPED
        state.notes.append(f"decode_failed: {exc}")
        return None


def _stage_separate(decode_result, data_dir, no_cache, state):
    state.status = TrackStatus.SEPARATING
    try:
        return separate_track(
            decode_result.track_id,
            decode_result.full_wav,
            data_dir=data_dir,
            no_cache=no_cache,
        )
    except SeparateAbortError:
        raise
    except SeparateSkipError as exc:
        logger.warning("[%s] [separate] Skipping: %s", state.short_id, exc)
        state.status = TrackStatus.SKIPPED
        state.notes.append(f"separate_failed: {exc}")
        return None


def _stage_analyse(stems, db_path, no_cache, state):
    state.status = TrackStatus.ANALYSING
    if not no_cache and cache.track_in_db(state.track_id, db_path):
        logger.info("[%s] [analysis] Cached, skipping", state.short_id)
        return
    analysis_runner.run_analysis(stems, db_path, state)


def _stage_transcribe(stems, db_path, no_cache, state):
    state.status = TrackStatus.TRANSCRIBING
    if not no_cache and cache.transcript_in_db(state.track_id, db_path):
        logger.info("[%s] [whisper] Cached, skipping", state.short_id)
        return
    try:
        transcribe_track(state.track_id, stems.vocals, db_path=db_path)
    except Exception as exc:
        logger.exception("[%s] [whisper] Failed: %s", state.short_id, exc)
        state.failed_stages.append("whisper")


def _stage_align(track_id, db_path, no_cache, state):
    state.status = TrackStatus.ALIGNING
    if not no_cache and cache.sections_labelled(track_id, db_path):
        logger.info("[%s] [alignment] Cached, skipping", state.short_id)
        return
    try:
        align_sections(track_id, db_path=db_path)
    except Exception as exc:
        logger.exception("[%s] [alignment] Failed: %s", state.short_id, exc)
        state.failed_stages.append("alignment")


def _stage_embed(track_id, db_path, use_qdrant, no_cache, state):
    state.status = TrackStatus.EMBEDDING
    if not use_qdrant:
        return
    if not no_cache and cache.vector_in_qdrant(track_id, db_path):
        logger.info("[%s] [embedding] Cached, skipping", state.short_id)
        return
    embed_track(track_id, db_path)


def _stage_score(track_id, mode, db_path, no_cache, state):
    state.status = TrackStatus.SCORING
    if not no_cache and cache.scores_exist(track_id, mode, db_path):
        logger.info("[%s] [scoring] Cached, skipping", state.short_id)
        return
    scoring_runner.run_scoring(track_id, mode, db_path, state)


def _build_summary(states: list[TrackState]) -> RunSummary:
    summary = RunSummary()
    for s in states:
        if s.status == TrackStatus.COMPLETE:
            summary.complete.append(s)
        elif s.status == TrackStatus.SKIPPED:
            summary.skipped.append(s)
        else:
            summary.partial.append(s)
    return summary
