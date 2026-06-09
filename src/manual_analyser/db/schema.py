"""
db/schema.py — SQLite schema definition and migrations.

Contains:
  SCHEMA      — full CREATE TABLE / CREATE INDEX DDL, applied on first run
  MIGRATIONS  — list of SQL strings applied once each, in order

To add a new column or table:
  1. Add a migration SQL string to MIGRATIONS
  2. Increment SCHEMA_VERSION
  3. Migrations must be idempotent (IF NOT EXISTS / IF EXISTS guards)

The SCHEMA itself is only applied to brand-new databases.
Existing databases are updated via MIGRATIONS only.
"""

from pathlib import Path

# Default DB path — overridden by passing path= to get_connection()
DEFAULT_DB_PATH = Path("data/manual_analyser.db")

# Current schema version. Increment when adding a migration.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Initial schema
# ---------------------------------------------------------------------------
# Applied once to a new database via conn.executescript().
# All CREATE statements use IF NOT EXISTS so re-running is safe.

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    -- identity
    track_id              TEXT PRIMARY KEY,   -- MD5 hex digest of path+filename
    filename              TEXT NOT NULL,
    artist                TEXT,               -- null if filename non-conformant
    song_name             TEXT,               -- null if filename non-conformant

    -- timing
    duration              REAL NOT NULL,      -- seconds (physical unit)

    -- tempo  [written by analysis/tempo.py]
    bpm                   REAL,               -- beats per minute (physical unit)
    bpm_confidence        REAL,               -- 0.0–1.0
    time_signature        INTEGER,            -- 3 or 4
    tempo_stability       REAL,               -- 0.0–1.0; 1.0 = perfectly metronomic

    -- groove  [written by analysis/groove.py]
    danceability          REAL,               -- 0.0–1.0 (essentia or approximation)
    self_similarity_score REAL,               -- 0.0–1.0
    beat_regularity       REAL,               -- 0.0–1.0
    groove_consistency    REAL,               -- composite: beat_regularity * self_similarity
    repetition_score      REAL,               -- 0.0–1.0; chroma-based

    -- rhythm feel  [written by analysis/rhythm.py]
    groove_feel           TEXT,               -- "straight" | "swung" | "unclear"

    -- harmony  [written by analysis/harmony.py]
    key                   TEXT,               -- e.g. "C", "F#"
    mode                  TEXT,               -- "major" | "minor"
    key_confidence        REAL,               -- 0.0–1.0

    -- energy  [written by analysis/energy.py]
    loudness_db           REAL,               -- normalised 0.0–1.0 (from LUFS)
    dynamic_range_db      REAL,               -- normalised 0.0–1.0
    verse_chorus_delta    REAL,               -- normalised 0.0–1.0; 0.15 ≈ 3dB lift
    energy_shape          TEXT,               -- "building" | "flat" | "peaked" | "unclear"

    -- lyrics / hook  [written by transcription/whisper.py]
    unique_word_ratio     REAL,               -- 0.0–1.0; low = more repetitive
    hook_repetition_count INTEGER,
    hook_first_appearance REAL,               -- seconds (physical unit)
    hook_phrase           TEXT,               -- most repeated phrase

    -- embedding  [written by embedding/embed.py]
    feature_summary       TEXT,               -- human-readable text sent to embedder

    -- metadata
    analysis_timestamp    TEXT NOT NULL,      -- ISO 8601
    analysis_version      TEXT NOT NULL       -- semver of tool that produced this row
);

CREATE TABLE IF NOT EXISTS sections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id         TEXT NOT NULL REFERENCES tracks(track_id),
    position         INTEGER NOT NULL,        -- 0-indexed order within track

    -- timing (physical units)
    start            REAL NOT NULL,
    end              REAL NOT NULL,
    duration         REAL NOT NULL,

    -- label  [initially "unknown"; updated by analysis/structure.py alignment pass]
    label            TEXT NOT NULL DEFAULT 'unknown',
        -- "intro" | "verse" | "pre_chorus" | "chorus" | "breakdown"
        -- | "double_chorus" | "bridge" | "outro" | "unknown"
    label_confidence REAL NOT NULL DEFAULT 0.0,  -- 0.0–1.0
    label_source     TEXT NOT NULL DEFAULT 'acoustic',
        -- "acoustic" | "lyric" | "hybrid"

    -- content  [updated by alignment pass]
    mean_energy      REAL,                    -- normalised 0.0–1.0
    lyric_density    REAL,                    -- words per second, normalised 0.0–1.0
    repeated_phrase  TEXT                     -- most repeated phrase, or null
);

CREATE INDEX IF NOT EXISTS idx_sections_track
    ON sections(track_id, position);

CREATE TABLE IF NOT EXISTS chord_progressions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id   INTEGER NOT NULL REFERENCES sections(id),
    progression  TEXT NOT NULL,               -- e.g. "Am - G - F - C"
    chords_json  TEXT NOT NULL                -- JSON: [{start, end, chord}, ...]
);

CREATE TABLE IF NOT EXISTS beat_patterns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id         TEXT NOT NULL REFERENCES tracks(track_id),

    -- 16-character binary strings, modal bar pattern
    -- "1" = hit, "0" = rest, 16th-note (semiquaver) resolution
    -- four-on-the-floor kick example: "1000100010001000"
    kick_pattern     TEXT NOT NULL,
    snare_pattern    TEXT NOT NULL,
    hihat_pattern    TEXT NOT NULL,

    syncopation_score  REAL,                  -- 0.0–1.0
    rhythmic_density   REAL                   -- 0.0–1.0
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id  TEXT NOT NULL REFERENCES tracks(track_id),
    start     REAL NOT NULL,                  -- seconds (physical unit)
    end       REAL NOT NULL,                  -- seconds (physical unit)
    text      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transcript_track
    ON transcript_segments(track_id, start);

CREATE TABLE IF NOT EXISTS scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id     TEXT NOT NULL REFERENCES tracks(track_id),
    mode         TEXT NOT NULL,               -- "1988" | "contemporary" | "1920s_1930s"
    criterion_id TEXT NOT NULL,               -- matches id in criteria TOML
    score        REAL,                        -- 0.0–1.0
    reasoning    TEXT,                        -- LLM explanation; null for deterministic rules
    passed       INTEGER NOT NULL,            -- 1 | 0
    scored_at    TEXT NOT NULL                -- ISO 8601
);

CREATE INDEX IF NOT EXISTS idx_scores_track_mode
    ON scores(track_id, mode);

CREATE INDEX IF NOT EXISTS idx_scores_mode_criterion
    ON scores(mode, criterion_id);

CREATE TABLE IF NOT EXISTS tracks_timeseries (
    track_id         TEXT PRIMARY KEY REFERENCES tracks(track_id),
    rms_profile_json TEXT NOT NULL
    -- JSON array of floats, 0.0–1.0, sampled every 0.5 seconds
    -- ~360 values for a 3-minute track
);

CREATE TABLE IF NOT EXISTS track_vectors (
    track_id  TEXT PRIMARY KEY REFERENCES tracks(track_id),
    qdrant_id TEXT NOT NULL               -- UUID string
    -- Row absent if embedding stage was skipped (Qdrant unavailable)
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""

# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------
# Each entry is applied once in order, keyed by its 1-based index.
# Migrations must be idempotent (IF NOT EXISTS / IF EXISTS guards).

MIGRATIONS: list[str] = [
    # Migration 1 — initial schema seed (no-op after first run)
    "INSERT OR IGNORE INTO schema_version (version) VALUES (1);",
]
