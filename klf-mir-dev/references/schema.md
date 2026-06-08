# Schema Reference

## tracks table (key fields only)

| Field | Type | Unit | Notes |
|---|---|---|---|
| track_id | TEXT PK | — | MD5 of path+filename |
| bpm | REAL | BPM | physical unit; used in lte 135 threshold |
| duration | REAL | seconds | physical unit; used in lte 210 threshold |
| groove_feel | TEXT | — | "straight" \| "swung" \| "unclear" |
| verse_chorus_delta | REAL | normalised | 0.0–1.0; 0.15 ≈ 3dB |
| hook_first_appearance | REAL | seconds | physical unit; used in lte 30 threshold |
| key | TEXT | — | e.g. "C", "F#" |
| mode | TEXT | — | "major" \| "minor" |
| energy_shape | TEXT | — | "building" \| "flat" \| "peaked" \| "unclear" |

All other numeric feature fields are normalised 0.0–1.0.

**Critical naming**: field is `verse_chorus_delta` (no `_db` suffix).

## sections table

| Field | Type | Notes |
|---|---|---|
| label | TEXT | "intro"\|"verse"\|"pre_chorus"\|"chorus"\|"breakdown"\|"double_chorus"\|"bridge"\|"outro"\|"unknown" |
| label_confidence | REAL | 0.0–1.0 |
| label_source | TEXT | "acoustic"\|"lyric"\|"hybrid" |
| start / end / duration | REAL | seconds (physical) |
| mean_energy | REAL | normalised 0.0–1.0 |

## scores table

| Field | Type | Notes |
|---|---|---|
| mode | TEXT | "1988" \| "contemporary" \| "1920s_1930s" |
| criterion_id | TEXT | matches TOML id field |
| score | REAL | 0.0–1.0 |
| reasoning | TEXT | null for deterministic rules |
| passed | INTEGER | 0 or 1 |

## Qdrant payload fields

```json
{
  "track_id": "md5",
  "artist": "str",
  "song_name": "str",
  "bpm": 126.4,        // raw BPM (not normalised)
  "key": "C",
  "mode": "major",
  "groove_feel": "straight",
  "mode_scores": {"1988": 0.82}
}
```

## Normalisation reference

| Field | Raw range | Formula |
|---|---|---|
| loudness_db | -60 to 0 dBFS | (value + 60) / 60 |
| dynamic_range_db | 0 to 60 dB | value / 60 |
| verse_chorus_delta | 0 to 20 dB | value / 20 |
| lyric_density | 0 to ~5 w/s | min(value / 5, 1.0) |
| rhythmic_density | 0 to 4 onsets/beat | min(value / 4, 1.0) |
