from __future__ import annotations

import sqlite3

import pytest

from median import shelves, taste

HORROR = 1  # persona ids come from db/seed_personas.sql (shelf rows only exist once a catalog does)
FILM = 1


def test_stars_text_uses_half_stars() -> None:
    assert shelves.stars_text(None) == "☆☆☆☆☆"
    assert shelves.stars_text(1) == "½☆☆☆☆"
    assert shelves.stars_text(7) == "★★★½☆"
    assert shelves.stars_text(10) == "★★★★★"
    assert shelves.stars_text(2) == "★☆☆☆☆"


@pytest.mark.parametrize("bad", [0, 11, -1])
def test_stars_text_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError):
        shelves.stars_text(bad)


def test_set_shelf_adds_and_moves(mini_catalog, db: sqlite3.Connection):
    shelves.set_shelf(db, HORROR, FILM, "want")
    item = shelves.get_item(db, HORROR, FILM)
    assert item is not None and item.shelf == "want" and item.rating is None
    shelves.set_shelf(db, HORROR, FILM, "in_progress")
    assert shelves.get_item(db, HORROR, FILM).shelf == "in_progress"  # type: ignore[union-attr]
    assert len(shelves.list_shelf(db, HORROR)) == 1


def test_moving_a_work_keeps_its_rating(mini_catalog, db: sqlite3.Connection):
    shelves.set_shelf(db, HORROR, FILM, "finished")
    shelves.set_rating(db, HORROR, FILM, 8)
    shelves.set_shelf(db, HORROR, FILM, "favorite")
    item = shelves.get_item(db, HORROR, FILM)
    assert item is not None and (item.shelf, item.rating) == ("favorite", 8)


def test_rating_an_unshelved_work_files_it_as_finished(mini_catalog, db: sqlite3.Connection):
    shelves.set_rating(db, HORROR, FILM, 9)
    item = shelves.get_item(db, HORROR, FILM)
    assert item is not None and (item.shelf, item.rating) == ("finished", 9)


def test_rating_never_moves_a_shelved_work(mini_catalog, db: sqlite3.Connection):
    shelves.set_shelf(db, HORROR, FILM, "want")
    shelves.set_rating(db, HORROR, FILM, 4)
    assert shelves.get_item(db, HORROR, FILM).shelf == "want"  # type: ignore[union-attr]


def test_clearing_a_rating(mini_catalog, db: sqlite3.Connection):
    shelves.set_rating(db, HORROR, FILM, 9)
    shelves.set_rating(db, HORROR, FILM, None)
    item = shelves.get_item(db, HORROR, FILM)
    assert item is not None and item.rating is None and item.shelf == "finished"


def test_clearing_the_rating_of_an_unshelved_work_does_nothing(
    mini_catalog, db: sqlite3.Connection
):
    shelves.set_rating(db, HORROR, FILM, None)
    assert shelves.get_item(db, HORROR, FILM) is None


def test_remove(mini_catalog, db: sqlite3.Connection):
    shelves.set_rating(db, HORROR, FILM, 9)
    shelves.remove(db, HORROR, FILM)
    assert shelves.get_item(db, HORROR, FILM) is None
    shelves.remove(db, HORROR, FILM)  # removing twice is fine


@pytest.mark.parametrize("bad", [0, 11, -3])
def test_invalid_rating_is_rejected_and_nothing_is_written(
    mini_catalog, db: sqlite3.Connection, bad: int
):
    with pytest.raises(ValueError):
        shelves.set_rating(db, HORROR, FILM, bad)
    assert shelves.get_item(db, HORROR, FILM) is None


def test_invalid_shelf_persona_or_work_is_rejected(mini_catalog, db: sqlite3.Connection):
    with pytest.raises(ValueError):
        shelves.set_shelf(db, HORROR, FILM, "dnf")
    with pytest.raises(ValueError):
        shelves.set_shelf(db, 99, FILM, "want")
    with pytest.raises(ValueError):
        shelves.set_shelf(db, HORROR, 9999, "want")
    with pytest.raises(ValueError):
        shelves.list_shelf(db, HORROR, "dnf")


def test_shelves_are_per_persona(mini_catalog, db: sqlite3.Connection):
    shelves.set_shelf(db, 1, FILM, "want")
    shelves.set_shelf(db, 2, FILM, "favorite")
    assert shelves.get_item(db, 1, FILM).shelf == "want"  # type: ignore[union-attr]
    assert shelves.get_item(db, 2, FILM).shelf == "favorite"  # type: ignore[union-attr]
    assert shelves.get_item(db, 3, FILM) is None


def test_list_shelf_filters_and_orders_newest_first(mini_catalog, db: sqlite3.Connection):
    for work_id, stamp in ((1, "2026-01-01"), (2, "2026-01-03"), (3, "2026-01-02")):
        shelves.set_shelf(db, HORROR, work_id, "want")
        db.execute(
            "UPDATE shelf_items SET added_at = ? WHERE persona_id = ? AND work_id = ?",
            (stamp, HORROR, work_id),
        )
    shelves.set_shelf(db, HORROR, 4, "finished")
    assert [e.work.id for e in shelves.list_shelf(db, HORROR, "want")] == [2, 3, 1]
    assert [e.work.id for e in shelves.list_shelf(db, HORROR, "finished")] == [4]
    assert len(shelves.list_shelf(db, HORROR)) == 4
    assert shelves.list_shelf(db, HORROR, "favorite") == []


def test_shelf_counts_include_zeros(mini_catalog, db: sqlite3.Connection):
    shelves.set_shelf(db, HORROR, 1, "want")
    shelves.set_shelf(db, HORROR, 2, "want")
    shelves.set_shelf(db, HORROR, 3, "favorite")
    assert shelves.shelf_counts(db, HORROR) == {
        "want": 2,
        "in_progress": 0,
        "finished": 0,
        "favorite": 1,
    }


def test_writes_recompute_the_taste_vector(mini_catalog, db: sqlite3.Connection):
    assert taste.get_taste_vector(db, HORROR) is None
    shelves.set_rating(db, HORROR, 1, 10)
    shelves.set_rating(db, HORROR, 11, 10)
    assert taste.get_taste_vector(db, HORROR) is None  # two signals: not enough yet
    shelves.set_rating(db, HORROR, 2, 9)
    assert taste.get_taste_vector(db, HORROR) is not None
    shelves.remove(db, HORROR, 2)
    assert taste.get_taste_vector(db, HORROR) is None  # back to two


def test_writes_are_committed(mini_catalog, db: sqlite3.Connection, tmp_path):
    from median.db import connect

    shelves.set_rating(db, HORROR, FILM, 6)
    other = connect(tmp_path / "test.db")
    try:
        row = other.execute("SELECT rating FROM shelf_items WHERE work_id = ?", (FILM,)).fetchone()
    finally:
        other.close()
    assert row is not None and row["rating"] == 6
