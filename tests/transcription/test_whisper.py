"""
transcription/whisper.py — Stage 3b of the pipeline.

Responsibilities:
  - Transcribe the vocals stem using openai-whisper
  - Extract hook phrase (most repeated n-gram across the track)
  - Compute unique word ratio (proxy for lyric repetitiveness)
  - Compute hook first appearance time
  - Write transcript segments to SQLite
  - Write hook metadata to tracks table

Writes to SQLite:
  INSERT INTO transcript_segments (track_id, start, end, text)
  — one row per Whisper segment

  UPDATE tracks SET
    unique_word_ratio, hook_repetition_count,
    hook_first_appearance, hook_phrase
  WHERE track_id = ?

Error handling (per docs/ERROR_HANDLING.md):
  - Whisper model not downloaded → log error, skip transcription
  - CUDA/MPS OOM → retry on CPU once; if still fails, skip
  - No speech detected → write empty transcript, set hook fields to null
  - Unhandled exception → log error, skip transcription for this track
"""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manual_analyser.audio.device import get_torch_device
from manual_analyser.db import get_connection

logger = logging.getLogger(__name__)

# Default Whisper model — configurable via CLI
DEFAULT_MODEL = "large-v3"
FALLBACK_MODEL = "medium"

# Hook phrase n-gram size
HOOK_NGRAM = 3

# Minimum repetitions for a phrase to be considered a hook
MIN_HOOK_REPETITIONS = 2


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TranscriptSegment:
    """A single Whisper output segment."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Full transcription result for a track."""

    segments: list[TranscriptSegment]
    full_text: str
    language: str
    hook_phrase: str | None
    hook_repetition_count: int
    hook_first_appearance: float | None  # seconds
    unique_word_ratio: float  # 0.0–1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_track(
    track_id: str,
    vocals_wav: Path,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> TranscriptionResult | None:
    """
    Transcribe the vocals stem and write results to SQLite.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        vocals_wav: Path to the vocals stem WAV (output of separate stage).
        data_dir: Root data directory (default: "data/").
        db_path: Path to SQLite database. Defaults to data/manual_analyser.db.
        model_name: Whisper model to use (default: large-v3).

    Returns:
        TranscriptionResult on success, or None if transcription failed.
    """
    short_id = track_id[:8]
    data_dir = Path(data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"

    try:
        result = _run_transcription(vocals_wav, model_name, short_id)
    except _TranscriptionSkipError as e:
        logger.warning("[%s] [whisper] Skipping transcription: %s", short_id, e)
        _write_nulls(resolved_db, track_id, short_id)
        return None
    except Exception as e:
        logger.exception("[%s] [whisper] Transcription failed: %s", short_id, e, exc_info=True)
        _write_nulls(resolved_db, track_id, short_id)
        return None

    _write_result(resolved_db, track_id, result, short_id)
    logger.info(
        "[%s] [whisper] %d segments, hook='%s' (x%d at %.1fs), unique_ratio=%.2f",
        short_id,
        len(result.segments),
        result.hook_phrase or "(none)",
        result.hook_repetition_count,
        result.hook_first_appearance or 0.0,
        result.unique_word_ratio,
    )
    return result


# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class _TranscriptionSkipError(Exception):
    """Raised when transcription should be skipped for this track."""


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _run_transcription(
    vocals_wav: Path,
    model_name: str,
    short_id: str,
) -> TranscriptionResult:
    """
    Load Whisper and transcribe the vocals stem.

    Tries the preferred device first; retries on CPU if OOM.

    Args:
        vocals_wav: Path to vocals stem WAV.
        model_name: Whisper model name.
        short_id: For log messages.

    Returns:
        TranscriptionResult.

    Raises:
        _TranscriptionSkipError: on model load failure or persistent OOM.
    """
    try:
        import whisper
    except ImportError as e:
        raise _TranscriptionSkipError(f"openai-whisper not installed: {e}") from e

    device = get_torch_device()

    try:
        model = _load_model(whisper, model_name, device, short_id)
        raw = _transcribe_audio(model, vocals_wav, device, short_id)
    except _OOMError:
        if device == "cpu":
            raise _TranscriptionSkipError(f"OOM on CPU — cannot transcribe {vocals_wav.name}")
        logger.warning("[%s] [whisper] OOM on %s, retrying on CPU", short_id, device)
        try:
            import whisper as whisper2

            model = _load_model(whisper2, model_name, "cpu", short_id)
            raw = _transcribe_audio(model, vocals_wav, "cpu", short_id)
        except _OOMError:
            raise _TranscriptionSkipError("OOM on both GPU and CPU")

    return _process_result(raw, short_id)


def _load_model(whisper_module, model_name: str, device: str, short_id: str):
    """Load a Whisper model, falling back to 'medium' if the requested model fails."""
    try:
        logger.debug("[%s] [whisper] Loading model %s on %s", short_id, model_name, device)
        return whisper_module.load_model(model_name, device=device)
    except Exception as e:
        if model_name != FALLBACK_MODEL:
            logger.warning(
                "[%s] [whisper] Failed to load %s (%s), falling back to %s",
                short_id,
                model_name,
                e,
                FALLBACK_MODEL,
            )
            return whisper_module.load_model(FALLBACK_MODEL, device=device)
        raise _TranscriptionSkipError(f"Could not load Whisper model {model_name}: {e}") from e


def _transcribe_audio(model, vocals_wav: Path, device: str, short_id: str) -> dict:
    """
    Run Whisper transcription on the vocals WAV.

    Args:
        model: Loaded Whisper model.
        vocals_wav: Path to the vocals WAV.
        device: Device string.
        short_id: For log messages.

    Returns:
        Raw Whisper result dict.

    Raises:
        _OOMError: if an out-of-memory error occurs.
    """
    import torch

    try:
        logger.debug("[%s] [whisper] Transcribing on %s", short_id, device)
        result = model.transcribe(
            str(vocals_wav),
            language=None,  # auto-detect
            word_timestamps=False,  # segment-level timestamps are sufficient
            verbose=False,
        )
        return result
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if _is_oom(e):
            raise _OOMError(str(e)) from e
        raise


def _process_result(raw: dict, short_id: str) -> TranscriptionResult:
    """
    Extract segments, hook phrase, and lyric statistics from raw Whisper output.

    Args:
        raw: Raw Whisper result dict with 'segments', 'text', 'language'.
        short_id: For log messages.

    Returns:
        TranscriptionResult.
    """
    segments = [
        TranscriptSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"].strip(),
        )
        for seg in raw.get("segments", [])
        if seg.get("text", "").strip()
    ]

    full_text = raw.get("text", "").strip()
    language = raw.get("language", "unknown")

    # Compute hook and lyric statistics
    hook_phrase, hook_count, hook_first = _extract_hook(segments)
    unique_ratio = _compute_unique_word_ratio(full_text)

    logger.debug(
        "[%s] [whisper] lang=%s segments=%d words=%d",
        short_id,
        language,
        len(segments),
        len(full_text.split()),
    )

    return TranscriptionResult(
        segments=segments,
        full_text=full_text,
        language=language,
        hook_phrase=hook_phrase,
        hook_repetition_count=hook_count,
        hook_first_appearance=hook_first,
        unique_word_ratio=unique_ratio,
    )


# ---------------------------------------------------------------------------
# Hook extraction
# ---------------------------------------------------------------------------


def _extract_hook(
    segments: list[TranscriptSegment],
    ngram_size: int = HOOK_NGRAM,
) -> tuple[str | None, int, float | None]:
    """
    Find the most repeated phrase across all transcript segments.

    Uses n-gram counting across all segment text. The most repeated
    n-gram that appears at least MIN_HOOK_REPETITIONS times is the hook.

    Args:
        segments: Transcript segments with timestamps.
        ngram_size: Size of n-gram (default: 3 words).

    Returns:
        (hook_phrase, repetition_count, first_appearance_time)
        All None/0 if no hook found.
    """
    if not segments:
        return None, 0, None

    # Collect all n-grams with their first timestamp
    ngram_times: dict[str, float] = {}
    ngram_counts: Counter = Counter()

    for seg in segments:
        words = seg.text.lower().split()
        if len(words) < ngram_size:
            continue

        for i in range(len(words) - ngram_size + 1):
            ngram = " ".join(words[i : i + ngram_size])
            # Record first appearance time
            if ngram not in ngram_times:
                ngram_times[ngram] = seg.start
            ngram_counts[ngram] += 1

    if not ngram_counts:
        return None, 0, None

    # Find most common phrase meeting minimum threshold
    most_common = ngram_counts.most_common(1)[0]
    phrase, count = most_common

    if count < MIN_HOOK_REPETITIONS:
        return None, 0, None

    first_time = ngram_times.get(phrase)
    return phrase, count, first_time


# ---------------------------------------------------------------------------
# Lyric statistics
# ---------------------------------------------------------------------------


def _compute_unique_word_ratio(text: str) -> float:
    """
    Compute the ratio of unique words to total words.

    Low ratio = high repetition = positive signal for The Manual criteria.

    Args:
        text: Full transcript text.

    Returns:
        Ratio 0.0–1.0. Returns 0.5 if text is empty (neutral).
    """
    if not text.strip():
        return 0.5

    words = text.lower().split()
    if not words:
        return 0.5

    unique_count = len(set(words))
    total_count = len(words)
    return float(np.clip(unique_count / total_count, 0.0, 1.0))


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------


class _OOMError(Exception):
    """Internal signal for out-of-memory conditions."""


def _is_oom(error: Exception) -> bool:
    """Return True if an exception represents an out-of-memory condition."""
    msg = str(error).lower()
    return "out of memory" in msg or "cuda out of memory" in msg or "mps backend out of memory" in msg


# ---------------------------------------------------------------------------
# SQLite writes
# ---------------------------------------------------------------------------


def _write_result(
    db_path: Path,
    track_id: str,
    result: TranscriptionResult,
    short_id: str,
) -> None:
    """Write transcript segments and hook metadata to SQLite."""
    conn = get_connection(db_path)
    try:
        with conn:
            # Insert transcript segments
            for seg in result.segments:
                conn.execute(
                    """
                    INSERT INTO transcript_segments (track_id, start, end, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (track_id, round(seg.start, 3), round(seg.end, 3), seg.text),
                )

            # Update tracks with hook metadata
            conn.execute(
                """
                UPDATE tracks SET
                    unique_word_ratio = ?,
                    hook_repetition_count = ?,
                    hook_first_appearance = ?,
                    hook_phrase = ?
                WHERE track_id = ?
                """,
                (
                    round(result.unique_word_ratio, 4),
                    result.hook_repetition_count,
                    round(result.hook_first_appearance, 3) if result.hook_first_appearance is not None else None,
                    result.hook_phrase,
                    track_id,
                ),
            )
    finally:
        conn.close()


def _write_nulls(db_path: Path, track_id: str, short_id: str) -> None:
    """Write null for all transcription fields when transcription fails."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE tracks SET
                    unique_word_ratio = NULL,
                    hook_repetition_count = NULL,
                    hook_first_appearance = NULL,
                    hook_phrase = NULL
                WHERE track_id = ?
                """,
                (track_id,),
            )
        logger.warning("[%s] [whisper] Wrote null fields due to failure", short_id)
    finally:
        conn.close()
