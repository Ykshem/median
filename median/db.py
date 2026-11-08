"""SQLite access: the only way to open the DB, schema setup and small query helpers.

Every connection goes through `connect()`, which loads sqlite-vec and turns on foreign keys and WAL.
Open one connection per Streamlit rerun; never share one across threads.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

from median.config import SCHEMA_PATH, SCHEMA_VERSION, SEED_PERSONAS_PATH, get_settings
from median.models import Persona

logger = logging.getLogger(__name__)


class SchemaVersionError(RuntimeError):
    """Raised when an existing DB was built with a different SCHEMA_VERSION."""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the Median DB (default: `MEDIAN_DB_PATH`) with sqlite-vec, FKs and WAL enabled.

    Creates the parent directory. Rows come back as `sqlite3.Row`.
    """
    db_path = Path(path) if path is not None else get_settings().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path | None = None, *, seed: bool = True) -> sqlite3.Connection:
    """Create or upgrade-in-place the schema and return an open connection.

    Idempotent. Seeds the demo personas unless `seed=False`. Raises SchemaVersionError if the file
    was built with a different SCHEMA_VERSION (rebuild or migrate it).
    """
    conn = connect(path)
    try:
        existing = get_meta(conn, "schema_version")
    except sqlite3.OperationalError:  # no meta table yet: a brand-new file
        existing = None
    if existing is not None and int(existing) != SCHEMA_VERSION:
        conn.close()
        raise SchemaVersionError(
            f"Database schema version is {existing} but this code expects {SCHEMA_VERSION}. "
            "Rebuild it with `uv run python -m pipeline.build --sample` or get a matching "
            "median.db from the team drive."
        )
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    if seed:
        conn.executescript(SEED_PERSONAS_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def ensure_db(path: str | Path | None = None) -> Path:
    """Create the DB (schema + personas) only if the file is missing; return its path.

    A shared `median.db` copied into `data/` is used untouched.
    """
    db_path = Path(path) if path is not None else get_settings().db_path
    if not db_path.exists():
        logger.info("No database at %s; creating an empty one", db_path)
        init_db(db_path).close()
    return db_path


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Return the value stored under `key` in the meta table, or None."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a meta key. Does not commit; the caller owns the transaction."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def list_personas(conn: sqlite3.Connection) -> list[Persona]:
    """Return all personas ordered by id."""
    rows = conn.execute(
        "SELECT id, name, description, taste_vector FROM personas ORDER BY id"
    ).fetchall()
    return [Persona(r["id"], r["name"], r["description"], r["taste_vector"]) for r in rows]


def count_works(conn: sqlite3.Connection) -> int:
    """Return how many works are in the catalog."""
    return conn.execute("SELECT COUNT(*) AS n FROM works").fetchone()["n"]


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Rebuild the external-content works_fts index from `works`. Call after loading works."""
    conn.execute("INSERT INTO works_fts(works_fts) VALUES ('rebuild')")


def vec_to_blob(vector: list[float]) -> bytes:
    """Serialize a float list to the float32 blob that sqlite-vec expects."""
    return sqlite_vec.serialize_float32(vector)
