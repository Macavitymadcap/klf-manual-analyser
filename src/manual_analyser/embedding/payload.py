"""embedding/payload.py — Build the Qdrant point payload from track features."""

from manual_analyser.embedding.db_reader import TrackFeatures


def build_payload(features: TrackFeatures) -> dict:
    """Build the Qdrant payload dict for a track. Omits None values."""
    candidate = {
        "track_id": features.track_id,
        "artist": features.artist,
        "song_name": features.song_name,
        "bpm": features.bpm,
        "key": features.key,
        "mode": features.mode,
        "groove_feel": features.groove_feel,
        "energy_shape": features.energy_shape,
        "danceability": features.danceability,
    }
    return {k: v for k, v in candidate.items() if v is not None}
