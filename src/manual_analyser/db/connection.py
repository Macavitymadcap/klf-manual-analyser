"""
db/connection.py — SQLite connection factory.

All other modules import get_connection() from manual_analyser.db (the
package __init__), not from here directly.

Responsibilities:
  - Open and configure a SQLite connection (WAL, foreign keys, row_factory)
  - Apply the initial schema on first run
  - Run any pending migrations
"""

import sqlite3
from pathlib import Path

from manual_analyser.db.schema import DEFAULT_DB_PATH, MIGRATIONS, SCHEMA


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """
    Return a configured sqlite3 connection.

    Creates the database file and schema if they do not exist.
    Runs any pending migrations.

    Args:
        path: Path to the database file. Defaults to DEFAULT_DB_PATH.

    Returns:
        An open sqlite3.Connection with row_factory=sqlite3.Row.
        Caller is responsible for closing, or use as a context manager:
            with get_connection() as conn: ...
    """
    db_path = Path(path) if path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # WAL mode: better concurrent read performance and crash safety
    conn.execute("PRAGMA journal_mode=WAL;")

    # Enforce foreign key constraints at the connection level
    conn.execute("PRAGMA foreign_keys=ON;")

    # Apply initial schema on first run (all statements are IF NOT EXISTS)
    conn.executescript(SCHEMA)
    conn.commit()

    # Apply any outstanding migrations
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
