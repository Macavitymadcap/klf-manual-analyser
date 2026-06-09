"""
audio/decode.py — Stage 1 of the pipeline.

Responsibilities:
  - Check ffmpeg is available (hard abort if not)
  - Decode MP3 to normalised mono WAV at 44100 Hz using ffmpeg
  - Parse artist and song title from the filename convention
  - Write the initial tracks row to SQLite
  - Return the stem directory path for downstream stages

Error handling (per docs/ERROR_HANDLING.md):
  - ffmpeg missing → raise DecodeAbortError (hard abort, stops entire run)
  - File not found → raise DecodeSkipError (per-track skip)
  - ffmpeg non-zero exit → raise DecodeSkipError (per-track skip)
  - Filename non-conformant → accept, store artist=None, emit warning
"""

import hashlib
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from manual_analyser.db import get_connection

logger = logging.getLogger(__name__)

# Current tool version — written to tracks.analysis_version
TOOL_VERSION = "0.1.0"

# Output audio format
SAMPLE_RATE = 44100
CHANNELS = 1  # mono

# Expected filename format: Artist_Name-Song_Title.mp3
_FILENAME_PATTERN = re.compile(r"^(?P<artist>[^-]+)-(?P<title>.+)$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DecodeAbortError(Exception):
    """
    Fatal error that should abort the entire pipeline run.
    Raised when ffmpeg is not found on PATH.
    """


class DecodeSkipError(Exception):
    """
    Per-track error — this track should be skipped, but other tracks continue.
    Raised when the input file is missing, unreadable, or ffmpeg fails.
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DecodeResult:
    """Returned by decode_track() on success."""

    track_id: str
    filename: str
    artist: str | None
    song_name: str | None
    duration: float  # seconds
    stem_dir: Path  # data/stems/{track_id}/
    full_wav: Path  # data/stems/{track_id}/full.wav


# ---------------------------------------------------------------------------
# Track identity and filename parsing (ingest helpers)
# ---------------------------------------------------------------------------


def make_track_id(path: Path | str) -> str:
    """
    Return a stable 32-character MD5 hex digest identifying a track.

    The digest is computed from the absolute path string, so the same file
    at the same location always produces the same ID. Moving or renaming
    the file produces a different ID — this is intentional, as the pipeline
    caches stems at data/stems/{track_id}/ and a renamed file should be
    treated as a new track.

    Args:
        path: Path to the MP3 file.

    Returns:
        32-character lowercase hex string.
    """
    abs_path = str(Path(path).resolve())
    return hashlib.md5(abs_path.encode()).hexdigest()


def parse_filename(path: Path | str) -> tuple[str | None, str | None]:
    """
    Parse artist and song title from a filename following the convention:
        Artist_Name-Song_Title.mp3

    Underscores separate words within artist or title. The hyphen separates
    artist from title (split on first hyphen only).

    Returns:
        (artist, song_name) — both title-cased with underscores replaced
        by spaces. Returns (None, None) if the filename does not match
        the expected pattern; the caller should log a warning.

    Examples:
        >>> parse_filename("The_KLF-Doctorin_The_Tardis.mp3")
        ('The Klf', 'Doctorin The Tardis')
        >>> parse_filename("unknown_file.mp3")
        (None, None)
    """
    stem = Path(path).stem
    match = _FILENAME_PATTERN.match(stem)
    if not match:
        return None, None

    artist = match.group("artist").replace("_", " ").title()
    title = match.group("title").replace("_", " ").title()
    return artist, title


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ffmpeg availability check
# ---------------------------------------------------------------------------


def check_ffmpeg() -> None:
    """
    Verify ffmpeg is available on PATH.

    Raises:
        DecodeAbortError: if ffmpeg is not found. The pipeline should
            catch this and abort the entire run with a helpful message.
    """
    if shutil.which("ffmpeg") is None:
        raise DecodeAbortError(
            "ffmpeg not found on PATH.\n"
            "Install it before running the pipeline:\n"
            "  Fedora:  sudo dnf install ffmpeg\n"
            "  macOS:   brew install ffmpeg"
        )


# ---------------------------------------------------------------------------
# Core decode function
# ---------------------------------------------------------------------------


def decode_track(
    mp3_path: Path | str,
    data_dir: Path | str = Path("data"),
    db_path: Path | str | None = None,
    no_cache: bool = False,
) -> DecodeResult:
    """
    Decode an MP3 file to a normalised mono WAV and register it in SQLite.

    If the output WAV already exists and no_cache is False, the decode step
    is skipped and the cached result is returned. The SQLite row is always
    written (or verified to exist).

    Args:
        mp3_path: Path to the input MP3 file.
        data_dir: Root data directory (default: "data/").
        db_path: Path to the SQLite database. Defaults to data/manual_analyser.db.
        no_cache: If True, re-decode even if WAV already exists.

    Returns:
        DecodeResult with track_id, metadata, duration, and paths.

    Raises:
        DecodeAbortError: if ffmpeg is not on PATH.
        DecodeSkipError: if the input file does not exist or ffmpeg fails.
    """
    mp3_path = Path(mp3_path)
    data_dir = Path(data_dir)

    if not mp3_path.exists():
        raise DecodeSkipError(f"File not found: {mp3_path}")

    if not mp3_path.suffix.lower() == ".mp3":
        raise DecodeSkipError(f"Expected .mp3 file, got: {mp3_path.suffix}")

    check_ffmpeg()

    track_id = make_track_id(mp3_path)
    short_id = track_id[:8]
    stem_dir = data_dir / "stems" / track_id
    full_wav = stem_dir / "full.wav"

    artist, song_name = parse_filename(mp3_path)
    if artist is None:
        logger.warning(
            "[%s] [decode] Filename does not match Artist_Name-Song_Title.mp3 "
            "convention: %s — storing with artist=null",
            short_id,
            mp3_path.name,
        )

    if full_wav.exists() and not no_cache:
        logger.info("[%s] [decode] WAV exists, skipping ffmpeg: %s", short_id, mp3_path.name)
    else:
        stem_dir.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(mp3_path, full_wav, short_id)
        logger.info("[%s] [decode] Decoded: %s", short_id, mp3_path.name)

    duration = _get_duration(full_wav, short_id)

    resolved_db = Path(db_path) if db_path else data_dir / "manual_analyser.db"
    _write_track_row(
        db_path=resolved_db,
        track_id=track_id,
        filename=mp3_path.name,
        artist=artist,
        song_name=song_name,
        duration=duration,
    )

    return DecodeResult(
        track_id=track_id,
        filename=mp3_path.name,
        artist=artist,
        song_name=song_name,
        duration=duration,
        stem_dir=stem_dir,
        full_wav=full_wav,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_ffmpeg(mp3_path: Path, output_wav: Path, short_id: str) -> None:
    """
    Run ffmpeg to decode mp3_path to a normalised mono WAV at 44100 Hz.

    Args:
        mp3_path: Input MP3.
        output_wav: Output WAV path.
        short_id: First 8 chars of track_id for log messages.

    Raises:
        DecodeSkipError: if ffmpeg returns a non-zero exit code.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp3_path),
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-af",
        "loudnorm",
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]

    logger.debug("[%s] [decode] ffmpeg command: %s", short_id, " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("[%s] [decode] ffmpeg failed (exit %d):\n%s", short_id, result.returncode, result.stderr)
        if output_wav.exists():
            output_wav.unlink()
        raise DecodeSkipError(f"ffmpeg failed for {mp3_path.name} (exit code {result.returncode})")


def _get_duration(wav_path: Path, short_id: str) -> float:
    """
    Get the duration of a WAV file in seconds using ffprobe.

    Falls back to reading the WAV header directly if ffprobe fails.

    Args:
        wav_path: Path to the WAV file.
        short_id: First 8 chars of track_id for log messages.

    Returns:
        Duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "compact=print_section=0:nokey=1:escape=csv",
        "-show_entries",
        "format=duration",
        str(wav_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass

    logger.warning("[%s] [decode] ffprobe failed, falling back to WAV header for duration", short_id)
    return _duration_from_wav_header(wav_path)


def _duration_from_wav_header(wav_path: Path) -> float:
    """
    Read duration from WAV file header without loading audio into memory.

    Returns:
        Duration in seconds, or 0.0 if the header cannot be read.
    """
    try:
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception as e:
        logger.warning("Could not read WAV header for duration: %s", e)
        return 0.0


def _write_track_row(
    db_path: Path,
    track_id: str,
    filename: str,
    artist: str | None,
    song_name: str | None,
    duration: float,
) -> None:
    """
    Insert the tracks row for this track_id.

    Uses INSERT OR IGNORE so re-running decode on a cached track does not
    overwrite analysis data written by later pipeline stages.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tracks (
                    track_id, filename, artist, song_name, duration,
                    analysis_timestamp, analysis_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (track_id, filename, artist, song_name, duration, utc_now_iso(), TOOL_VERSION),
            )
    finally:
        conn.close()
