# RMS profile sampling interval in seconds
from dataclasses import dataclass

RMS_SAMPLE_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnergyResult:
    """Energy analysis results for a single track."""

    loudness_db: float  # normalised 0.0–1.0 (from LUFS)
    dynamic_range_db: float  # normalised 0.0–1.0
    verse_chorus_delta: float  # normalised 0.0–1.0; 0.15 ≈ 3dB
    energy_shape: str  # "building" | "flat" | "peaked" | "unclear"
    rms_profile: list[float]  # normalised RMS values, one per 0.5s
