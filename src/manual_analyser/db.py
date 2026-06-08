"""
db.py — SQLite connection, schema definition, and migrations.

All other modules import `get_connection()` from here. Schema changes
are handled via the `MIGRATIONS` list; each migration is a SQL string
applied exactly once, identified by its index.

Usage:
    from manual_analyser.db import get_connection
    with get_connection() as conn:
        conn.execute("SELECT * FROM tracks")
"""

import sqlite3
from pathlib import Path

# Default DB path — can be overridden via get_connection(path=...)
DEFAULT_DB_PATH = Path("data/manual_analyser.db")

# Current schema version. Increment when adding a migration.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Each string is a complete CREATE TABLE / CREATE INDEX statement.
# Statements are applied in order on first run.
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
    score        REAL NOT NULL,               -- 0.0–1.0
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
# To add a migration: append a SQL string and increment SCHEMA_VERSION above.
# Migrations must be idempotent (use IF NOT EXISTS / IF EXISTS guards).

MIGRATIONS: list[str] = [
    # Migration 1 — initial schema (applied on first run, no-op thereafter)
    # The full schema above handles this; this entry exists to seed the version table.
    "INSERT OR IGNORE INTO schema_version (version) VALUES (1);",
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """
    Return a sqlite3 connection with WAL mode, foreign keys enabled,
    and row_factory set to sqlite3.Row for dict-like row access.

    Creates the database file and schema if it does not exist.
    Runs any pending migrations.

    Args:
        path: Path to the database file. Defaults to DEFAULT_DB_PATH.

    Returns:
        An open sqlite3.Connection. Caller is responsible for closing,
        or use as a context manager: `with get_connection() as conn: ...`
    """
    db_path = Path(path) if path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Enable WAL for better concurrent read performance and crash safety
    conn.execute("PRAGMA journal_mode=WAL;")

    # Enforce foreign key constraints
    conn.execute("PRAGMA foreign_keys=ON;")

    # Create schema on first run
    conn.executescript(SCHEMA)
    conn.commit()

    # Run pending migrations
    _run_migrations(conn)

    return conn


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply any migrations not yet recorded in schema_version."""
    current = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0

    for i, migration_sql in enumerate(MIGRATIONS, start=1):
        if i > current:
            conn.executescript(migration_sql)
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (?)", (i,))
            conn.commit()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def track_exists(conn: sqlite3.Connection, track_id: str) -> bool:
    """Return True if a track row exists for the given track_id."""
    row = conn.execute("SELECT 1 FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
    return row is not None


def transcript_exists(conn: sqlite3.Connection, track_id: str) -> bool:
    """Return True if any transcript segments exist for this track."""
    row = conn.execute("SELECT 1 FROM transcript_segments WHERE track_id = ? LIMIT 1", (track_id,)).fetchone()
    return row is not None


def sections_labelled(conn: sqlite3.Connection, track_id: str) -> bool:
    """
    Return True if the alignment pass has run for this track —
    i.e. at least one section has a label other than 'unknown'.
    """
    row = conn.execute(
        "SELECT 1 FROM sections WHERE track_id = ? AND label != 'unknown' LIMIT 1", (track_id,)
    ).fetchone()
    return row is not None


def vector_exists(conn: sqlite3.Connection, track_id: str) -> bool:
    """Return True if a Qdrant vector record exists for this track."""
    row = conn.execute("SELECT 1 FROM track_vectors WHERE track_id = ?", (track_id,)).fetchone()
    return row is not None


def scores_exist(conn: sqlite3.Connection, track_id: str, mode: str) -> bool:
    """Return True if scoring has been run for this track + mode combination."""
    row = conn.execute("SELECT 1 FROM scores WHERE track_id = ? AND mode = ? LIMIT 1", (track_id, mode)).fetchone()
    return row is not None


def get_section_sequence(conn: sqlite3.Connection, track_id: str) -> list[str]:
    """Return the ordered list of section labels for a track."""
    rows = conn.execute("SELECT label FROM sections WHERE track_id = ? ORDER BY position", (track_id,)).fetchall()
    return [row["label"] for row in rows]


def get_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    """Return the full tracks row for a track_id, or None if not found."""
    return conn.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,)).fetchone()


def get_all_track_ids(conn: sqlite3.Connection) -> list[str]:
    """Return all track_ids in the database."""
    rows = conn.execute("SELECT track_id FROM tracks").fetchall()
    return [row["track_id"] for row in rows]
