"""
db — SQLite database package.

All other modules continue to import from manual_analyser.db exactly as before:

    from manual_analyser.db import get_connection
    from manual_analyser.db import get_connection, track_exists, get_track

Sub-modules:
    schema.py     — SCHEMA DDL string and MIGRATIONS list
    connection.py — get_connection() factory and _run_migrations()
    helpers.py    — convenience query functions (track_exists, get_track, …)
"""

from manual_analyser.db.connection import DEFAULT_DB_PATH, get_connection
from manual_analyser.db.helpers import (
    get_all_track_ids,
    get_section_sequence,
    get_track,
    scores_exist,
    sections_labelled,
    track_exists,
    transcript_exists,
    vector_exists,
)
from manual_analyser.db.schema import MIGRATIONS, SCHEMA, SCHEMA_VERSION

__all__ = [
    # Connection
    "get_connection",
    "DEFAULT_DB_PATH",
    # Schema
    "SCHEMA",
    "SCHEMA_VERSION",
    "MIGRATIONS",
    # Helpers
    "track_exists",
    "transcript_exists",
    "sections_labelled",
    "vector_exists",
    "scores_exist",
    "get_section_sequence",
    "get_track",
    "get_all_track_ids",
]
