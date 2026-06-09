"""embedding/summarise.py — Build human-readable feature summary from track features."""

from manual_analyser.embedding.db_reader import TrackFeatures


def build_summary(features: TrackFeatures) -> str:
    """Assemble a human-readable text summary of a track's features."""
    lines = [
        _identity_line(features),
        _tempo_line(features),
        _harmony_line(features),
        _groove_line(features),
        _structure_line(features),
        _lyrics_line(features),
        _rhythm_line(features),
    ]
    return "\n".join(line for line in lines if line)


def _identity_line(f: TrackFeatures) -> str:
    if f.artist and f.song_name:
        return f"Track: {f.artist} — {f.song_name}"
    return f"Track ID: {f.track_id[:8]}"


def _tempo_line(f: TrackFeatures) -> str:
    if f.bpm is None:
        return ""
    return f"Tempo: {f.bpm:.1f} BPM"


def _harmony_line(f: TrackFeatures) -> str:
    if not f.key:
        return ""
    mode = f.mode or "unknown mode"
    return f"Key: {f.key} {mode}"


def _groove_line(f: TrackFeatures) -> str:
    parts = []
    if f.groove_feel:
        parts.append(f"groove={f.groove_feel}")
    if f.danceability is not None:
        parts.append(f"danceability={f.danceability:.2f}")
    if f.energy_shape:
        parts.append(f"energy_shape={f.energy_shape}")
    return f"Groove: {', '.join(parts)}" if parts else ""


def _structure_line(f: TrackFeatures) -> str:
    if not f.section_labels:
        return ""
    return f"Structure: {' → '.join(f.section_labels)}"


def _lyrics_line(f: TrackFeatures) -> str:
    parts = []
    if f.hook_phrase:
        count = f.hook_repetition_count or 0
        parts.append(f'hook="{f.hook_phrase}" (x{count})')
    if f.unique_word_ratio is not None:
        parts.append(f"unique_word_ratio={f.unique_word_ratio:.2f}")
    return f"Lyrics: {', '.join(parts)}" if parts else ""


def _rhythm_line(f: TrackFeatures) -> str:
    if not f.kick_pattern:
        return ""
    snare = f.snare_pattern or "?" * 16
    return f"Rhythm: kick={f.kick_pattern} snare={snare}"
