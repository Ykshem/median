from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from median import config
from median.db import (
    SchemaVersionError,
    connect,
    count_works,
    ensure_db,
    get_meta,
    init_db,
    list_personas,
    rebuild_fts,
    set_meta,
    vec_to_blob,
)


def test_connect_sets_pragmas_and_loads_sqlite_vec(tmp_path: Path) -> None:
    conn = connect(tmp_path / "x.db")
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("SELECT vec_version()").fetchone()[0].startswith("v0.1")
        assert conn.row_factory is sqlite3.Row
    finally:
        conn.close()


def test_init_db_is_idempotent_and_seeds_personas(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    init_db(path).close()
    conn = init_db(path)
    try:
        assert [p.name for p in list_personas(conn)][0] == "Horror fan"
        assert len(list_personas(conn)) == 4
        assert get_meta(conn, "schema_version") == str(config.SCHEMA_VERSION)
    finally:
        conn.close()


def test_init_db_without_seed_has_no_personas(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "x.db", seed=False)
    try:
        assert list_personas(conn) == []
    finally:
        conn.close()


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    conn = init_db(path)
    set_meta(conn, "schema_version", "999")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError, match="999"):
        init_db(path)


def test_ensure_db_creates_missing_file_only(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "new.db"
    assert ensure_db(path) == path
    assert path.exists()
    conn = connect(path)
    try:
        assert len(list_personas(conn)) == 4
    finally:
        conn.close()


def test_check_constraint_rejects_bad_media_type(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO works (media_type, title) VALUES ('podcast', 'x')")


def test_foreign_keys_are_enforced(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO shelf_items (persona_id, work_id, shelf) VALUES (999, 999, 'want')")


def test_rating_must_be_between_1_and_10(db: sqlite3.Connection) -> None:
    db.execute("INSERT INTO works (id, media_type, title) VALUES (1, 'movie', 'A')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO shelf_items (persona_id, work_id, shelf, rating) "
            "VALUES (1, 1, 'finished', 11)"
        )


def _unit(i: int, dims: int = config.EMBED_DIMS) -> list[float]:
    vec = [0.0] * dims
    vec[i] = 1.0
    return vec


def test_knn_cosine_distance_and_media_type_filter(db: sqlite3.Connection) -> None:
    rows = [(1, _unit(0), "movie"), (2, _unit(0), "book"), (3, _unit(1), "book")]
    for work_id, vec, media in rows:
        db.execute(
            "INSERT INTO vec_vibes (work_id, embedding, media_type) VALUES (?, ?, ?)",
            (work_id, vec_to_blob(vec), media),
        )
    query = vec_to_blob(_unit(0))
    sql = (
        "SELECT work_id, distance FROM vec_vibes "
        "WHERE embedding MATCH ? AND k = 3 {extra} ORDER BY distance"
    )
    everything = db.execute(sql.format(extra=""), (query,)).fetchall()
    assert {r["work_id"] for r in everything[:2]} == {1, 2}
    assert math.isclose(everything[0]["distance"], 0.0, abs_tol=1e-6)
    assert math.isclose(everything[-1]["distance"], 1.0, abs_tol=1e-6)  # orthogonal

    books = db.execute(sql.format(extra="AND media_type = 'book'"), (query,)).fetchall()
    assert [r["work_id"] for r in books] == [2, 3]


def test_fts_finds_titles_and_creators_after_rebuild(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT INTO works (id, media_type, title, creators) "
        "VALUES (1, 'book', 'The Haunting of Hill House', '[\"Shirley Jackson\"]')"
    )
    rebuild_fts(db)
    hits = db.execute("SELECT rowid FROM works_fts WHERE works_fts MATCH 'haunting'").fetchall()
    assert [h["rowid"] for h in hits] == [1]
    by_author = db.execute("SELECT rowid FROM works_fts WHERE works_fts MATCH 'jackson'").fetchall()
    assert [h["rowid"] for h in by_author] == [1]
    assert count_works(db) == 1
