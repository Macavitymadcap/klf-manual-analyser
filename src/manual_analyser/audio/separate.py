"""
audio/separate.py — Stage 2 of the pipeline.

Responsibilities:
  - Separate a mono WAV into four stems using Demucs (htdemucs model):
    drums.wav, bass.wav, vocals.wav, other.wav
  - Use the best available device (cuda > mps > cpu)
  - Retry on CPU if GPU runs out of memory
  - Write stems to data/stems/{track_id}/

Error handling (per docs/ERROR_HANDLING.md):
  - Demucs model not found → raise SeparateAbortError (hard abort)
  - CUDA OOM → retry on CPU; if CPU also fails → raise SeparateSkipError
  - Incomplete stems produced → raise SeparateSkipError
  - Unhandled exception → raise SeparateSkipError
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from manual_analyser.audio.device import get_torch_device

logger = logging.getLogger(__name__)

# Expected stem filenames produced by Demucs
STEM_NAMES = ("drums", "bass", "vocals", "other")

# Demucs model to use
DEMUCS_MODEL = "htdemucs"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SeparateAbortError(Exception):
    """
    Fatal error that should abort the entire pipeline run.
    Raised when the Demucs model cannot be loaded.
    """


class SeparateSkipError(Exception):
    """
    Per-track error — this track should be skipped, other tracks continue.
    Raised on OOM (after CPU retry), incomplete output, or unhandled exception.
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SeparateResult:
    """Returned by separate_track() on success."""

    track_id: str
    stem_dir: Path
    drums: Path
    bass: Path
    vocals: Path
    other: Path
    device_used: str  # "cuda", "mps", or "cpu"
    was_cpu_fallback: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def separate_track(
    track_id: str,
    full_wav: Path,
    data_dir: Path | str = Path("data"),
    no_cache: bool = False,
) -> SeparateResult:
    """
    Separate full_wav into four stems using Demucs (htdemucs).

    If all four stem files already exist and no_cache is False, separation
    is skipped and the cached result is returned.

    Args:
        track_id: 32-char MD5 hex digest identifying the track.
        full_wav: Path to the decoded mono WAV (output of decode stage).
        data_dir: Root data directory (default: "data/").
        no_cache: If True, re-separate even if stems already exist.

    Returns:
        SeparateResult with paths to all four stem WAVs.

    Raises:
        SeparateAbortError: if the Demucs model cannot be loaded.
        SeparateSkipError: if separation fails after retries.
    """
    data_dir = Path(data_dir)
    stem_dir = data_dir / "stems" / track_id
    short_id = track_id[:8]

    if not no_cache and _stems_exist(stem_dir):
        logger.info("[%s] [separate] Stems exist, skipping Demucs", short_id)
        return _build_result(track_id, stem_dir, device_used="cached", was_cpu_fallback=False)

    stem_dir.mkdir(parents=True, exist_ok=True)
    device = get_torch_device()

    logger.info("[%s] [separate] Running Demucs on device: %s", short_id, device)

    try:
        _run_demucs(full_wav, stem_dir, device, short_id)
    except _OOMError:
        if device == "cpu":
            raise SeparateSkipError(f"[{short_id}] Demucs OOM on CPU — track too large to separate")
        logger.warning("[%s] [separate] OOM on %s, retrying on CPU", short_id, device)
        try:
            _run_demucs(full_wav, stem_dir, "cpu", short_id)
            device = "cpu"
        except _OOMError:
            raise SeparateSkipError(f"[{short_id}] Demucs OOM on both {device} and CPU")

    missing = [name for name in STEM_NAMES if not (stem_dir / f"{name}.wav").exists()]
    if missing:
        raise SeparateSkipError(f"[{short_id}] Demucs produced incomplete stems, missing: {missing}")

    was_fallback = device == "cpu" and get_torch_device() != "cpu"
    logger.info("[%s] [separate] Done (device: %s)", short_id, device)

    return _build_result(track_id, stem_dir, device_used=device, was_cpu_fallback=was_fallback)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _stems_exist(stem_dir: Path) -> bool:
    """Return True if all four expected stem WAVs exist in stem_dir."""
    return all((stem_dir / f"{name}.wav").exists() for name in STEM_NAMES)


def _build_result(
    track_id: str,
    stem_dir: Path,
    device_used: str,
    was_cpu_fallback: bool,
) -> SeparateResult:
    """Construct a SeparateResult from a stem directory."""
    return SeparateResult(
        track_id=track_id,
        stem_dir=stem_dir,
        drums=stem_dir / "drums.wav",
        bass=stem_dir / "bass.wav",
        vocals=stem_dir / "vocals.wav",
        other=stem_dir / "other.wav",
        device_used=device_used,
        was_cpu_fallback=was_cpu_fallback,
    )


class _OOMError(Exception):
    """Internal signal for out-of-memory conditions during Demucs."""


def _is_oom(error: Exception) -> bool:
    """Return True if an exception represents an out-of-memory condition."""
    msg = str(error).lower()
    return "out of memory" in msg or "cuda out of memory" in msg or "mps backend out of memory" in msg or "oom" in msg


def _run_demucs(
    full_wav: Path,
    stem_dir: Path,
    device: str,
    short_id: str,
) -> None:
    """
    Run Demucs htdemucs separation on full_wav, writing stems to stem_dir.

    Args:
        full_wav: Input WAV path.
        stem_dir: Output directory for stems.
        device: Torch device string ("cuda", "mps", or "cpu").
        short_id: First 8 chars of track_id for log messages.

    Raises:
        SeparateAbortError: if the model cannot be loaded.
        _OOMError: if a CUDA/MPS out-of-memory error occurs.
        SeparateSkipError: for other unhandled Demucs errors.
    """
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio
        from demucs.pretrained import get_model
    except ImportError as e:
        raise SeparateAbortError(f"Could not import Demucs: {e}\nEnsure demucs is installed: uv add demucs") from e

    try:
        model = get_model(DEMUCS_MODEL)
        model.eval()
    except Exception as e:
        raise SeparateAbortError(
            f"Could not load Demucs model '{DEMUCS_MODEL}': {e}\n"
            "Try: python -m demucs --help to verify the model is available."
        ) from e

    try:
        model.to(device)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if _is_oom(e):
            raise _OOMError(str(e)) from e
        raise SeparateSkipError(f"[{short_id}] Failed to move model to {device}: {e}") from e

    try:
        wav = AudioFile(full_wav).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
    except Exception as e:
        raise SeparateSkipError(f"[{short_id}] Could not load audio for separation: {e}") from e

    try:
        with torch.no_grad():
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            sources = apply_model(
                model,
                wav[None],
                device=device,
                progress=False,
                num_workers=0,
            )[0]
            sources = sources * ref.std() + ref.mean()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if _is_oom(e):
            raise _OOMError(str(e)) from e
        raise SeparateSkipError(f"[{short_id}] Demucs separation failed: {e}") from e
    except Exception as e:
        raise SeparateSkipError(f"[{short_id}] Unexpected error during separation: {e}") from e

    try:
        for stem_name, source in zip(model.sources, sources):
            out_path = stem_dir / f"{stem_name}.wav"
            save_audio(source, str(out_path), samplerate=model.samplerate)
            logger.debug("[%s] [separate] Saved stem: %s", short_id, out_path.name)
    except Exception as e:
        raise SeparateSkipError(f"[{short_id}] Failed to save stems: {e}") from e
