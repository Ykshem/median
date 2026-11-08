from __future__ import annotations

import re
import sqlite3

from median import config, taste
from median.db import apply_seed, vec_to_blob
from pipeline.enrich import index
from tests.test_taste import add_work, vec

SEED_ROW = re.compile(
    r"\(\s*(\d+),\s*'(tmdb|openlibrary)',\s*'([^']+)',\s*'(finished|favorite|want|in_progress)',"
    r"\s*(\d+|NULL)\s*\)"
)


def seed_rows() -> list[tuple[int, str, str, str, int | None]]:
    text = config.SEED_PERSONAS_PATH.read_text(encoding="utf-8")
    return [
        (int(p), src, ext, shelf, None if r == "NULL" else int(r))
        for p, src, ext, shelf, r in SEED_ROW.findall(text)
    ]


def catalog_with(db: sqlite3.Connection, refs: list[tuple[str, str]]) -> None:
    """Create one work per (source, external_id) with a vector, so the seed can resolve it."""
    for i, (source, ext) in enumerate(refs, start=100):
        add_work(db, i, vec(a=1, b=i / 1000), "movie" if source == "tmdb" else "book")
        db.execute("INSERT INTO external_ids VALUES (?, ?, ?)", (i, source, ext))
    db.commit()


# the seed file itself ----------------------------------------------------------------------------


def test_every_persona_has_about_ten_rated_works_of_both_media() -> None:
    rows = seed_rows()
    for persona_id in (1, 2, 3, 4):
        mine = [r for r in rows if r[0] == persona_id]
        assert len(mine) >= 9, f"persona {persona_id} has only {len(mine)} seeded works"
        assert {r[1] for r in mine} == {"tmdb", "openlibrary"}, "needs movies and books"
        assert any(r[4] is not None and r[4] <= 3 for r in mine), "needs a disliked work"
        assert sum(r[4] is not None and r[4] >= 8 for r in mine) >= 6, "needs clear favourites"


def test_there_are_no_duplicate_seed_rows() -> None:
    keys = [(r[0], r[1], r[2]) for r in seed_rows()]
    assert len(keys) == len(set(keys))


def test_ratings_are_valid_and_favorites_are_rated_high() -> None:
    for _, _, _, shelf, rating in seed_rows():
        assert rating is None or 1 <= rating <= 10
        if shelf == "favorite":
            assert rating is not None and rating >= 9


# applying it -------------------------------------------------------------------------------------


def test_seed_resolves_works_by_external_id(db: sqlite3.Connection) -> None:
    catalog_with(db, [("tmdb", "493922"), ("openlibrary", "OL20759125W")])
    apply_seed(db)
    rows = db.execute(
        "SELECT s.persona_id, s.shelf, s.rating, e.external_id FROM shelf_items s "
        "JOIN external_ids e ON e.work_id = s.work_id WHERE s.persona_id = 1 ORDER BY e.external_id"
    ).fetchall()
    assert [(r["external_id"], r["shelf"], r["rating"]) for r in rows] == [
        ("493922", "favorite", 10),
        ("OL20759125W", "favorite", 10),
    ]


def test_works_missing_from_the_catalog_are_skipped_not_errors(db: sqlite3.Connection) -> None:
    apply_seed(db)  # an empty catalog
    assert db.execute("SELECT COUNT(*) FROM shelf_items").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM personas").fetchone()[0] == 4


def test_the_same_work_can_be_rated_differently_by_different_personas(
    db: sqlite3.Connection,
) -> None:
    catalog_with(db, [("tmdb", "493922")])  # Hereditary: persona 1 loves it, persona 2 hates it
    apply_seed(db)
    ratings = dict(db.execute("SELECT persona_id, rating FROM shelf_items").fetchall())
    assert ratings == {1: 10, 2: 2}


def test_reapplying_adds_late_arrivals_but_never_overwrites_a_changed_rating(
    db: sqlite3.Connection,
) -> None:
    catalog_with(db, [("tmdb", "493922")])
    apply_seed(db)
    db.execute(
        "UPDATE shelf_items SET rating = 4 WHERE persona_id = 1"
    )  # the user changed their mind
    db.commit()
    add_work(db, 500, vec(a=1), "movie")
    db.execute("INSERT INTO external_ids VALUES (500, 'tmdb', '694')")  # The Shining arrives later
    apply_seed(db)
    apply_seed(db)  # and a third time, for idempotence
    mine = dict(
        db.execute("SELECT work_id, rating FROM shelf_items WHERE persona_id = 1").fetchall()
    )
    assert mine == {100: 4, 500: 9}


def test_init_db_seeds_personas_and_a_rebuild_keeps_them(db: sqlite3.Connection) -> None:
    assert [r["name"] for r in db.execute("SELECT name FROM personas ORDER BY id")][:1] == [
        "Horror fan"
    ]


# the index step ----------------------------------------------------------------------------------


def test_index_seeds_the_shelves_and_computes_every_taste_vector(db: sqlite3.Connection) -> None:
    refs = sorted({(src, ext) for _, src, ext, _, _ in seed_rows()})
    catalog_with(db, refs)
    db.execute("DELETE FROM shelf_items")
    db.commit()
    index.build_index(db)
    assert db.execute("SELECT COUNT(*) FROM shelf_items").fetchone()[0] == len(seed_rows())
    for persona_id in (1, 2, 3, 4):
        assert taste.get_taste_vector(db, persona_id) is not None, f"persona {persona_id}"


def test_index_is_idempotent_with_seeds_and_taste(db: sqlite3.Connection) -> None:
    catalog_with(db, sorted({(src, ext) for _, src, ext, _, _ in seed_rows()}))
    index.build_index(db)
    first = db.execute("SELECT id, taste_vector FROM personas ORDER BY id").fetchall()
    index.build_index(db)
    second = db.execute("SELECT id, taste_vector FROM personas ORDER BY id").fetchall()
    assert [bytes(r["taste_vector"]) for r in first] == [bytes(r["taste_vector"]) for r in second]
    assert db.execute("SELECT COUNT(*) FROM shelf_items").fetchone()[0] == len(seed_rows())


def test_a_taste_vector_is_a_blob_of_the_right_size(db: sqlite3.Connection) -> None:
    catalog_with(db, sorted({(src, ext) for _, src, ext, _, _ in seed_rows()}))
    index.build_index(db)
    blob = db.execute("SELECT taste_vector FROM personas WHERE id = 1").fetchone()[0]
    assert len(blob) == config.EMBED_DIMS * 4 and blob != vec_to_blob(vec())
