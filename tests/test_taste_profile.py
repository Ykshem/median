from __future__ import annotations

import sqlite3

from median import shelves, taste

HORROR = 1
HORROR_WORKS = [1, 2, 11, 12]  # The Shining and The Haunting, film and book
COZY_WORKS = [3, 4, 13]  # Knives Out, Orient Express (film), Thursday Murder Club


def test_profile_without_signals_is_empty(mini_catalog, db: sqlite3.Connection):
    p = taste.profile(db, HORROR)
    assert (p.signals, p.has_vector) == (0, False)
    assert p.themes == [] and p.moods == [] and p.nearest == []


def test_profile_below_the_signal_minimum_has_no_vector(mini_catalog, db: sqlite3.Connection):
    shelves.set_rating(db, HORROR, 1, 10)
    shelves.set_rating(db, HORROR, 2, 9)
    p = taste.profile(db, HORROR)
    assert p.signals == 2 and not p.has_vector and p.themes == []
    assert p.liked == [1, 2]


def test_profile_describes_the_taste_in_the_nearest_cards(mini_catalog, db: sqlite3.Connection):
    for work_id in HORROR_WORKS:
        shelves.set_rating(db, HORROR, work_id, 10)
    p = taste.profile(db, HORROR)
    assert p.has_vector and p.signals == 4
    assert p.nearest[0] in HORROR_WORKS
    assert set(HORROR_WORKS) <= set(p.nearest[:6])
    assert "eerie" in dict(p.moods)
    counts = [n for _, n in p.themes]
    assert counts == sorted(counts, reverse=True) and len(p.themes) <= 8


def test_profile_top_limits_the_term_lists(mini_catalog, db: sqlite3.Connection):
    for work_id in HORROR_WORKS:
        shelves.set_rating(db, HORROR, work_id, 10)
    p = taste.profile(db, HORROR, top=2)
    assert len(p.themes) <= 2 and len(p.moods) <= 2


def test_profile_separates_liked_from_disliked(mini_catalog, db: sqlite3.Connection):
    shelves.set_rating(db, HORROR, 1, 10)
    shelves.set_rating(db, HORROR, 2, 8)
    shelves.set_rating(db, HORROR, 3, 1)
    shelves.set_rating(db, HORROR, 13, 2)
    p = taste.profile(db, HORROR)
    assert p.liked == [1, 2]
    assert p.disliked == [3, 13]  # most disliked first


def test_signals_disliked_is_the_mirror_of_liked() -> None:
    s = taste.Signals({1: 0.5, 2: -0.2, 3: -0.9, 4: 0.0}, 4)
    assert s.liked() == [1]
    assert s.disliked() == [3, 2]


def test_profile_of_different_personas_differs(mini_catalog, db: sqlite3.Connection):
    for work_id in HORROR_WORKS:
        shelves.set_rating(db, 1, work_id, 10)
    for work_id in COZY_WORKS:
        shelves.set_rating(db, 2, work_id, 10)
    horror, cozy = taste.profile(db, 1), taste.profile(db, 2)
    assert "eerie" in dict(horror.moods) and "eerie" not in dict(cozy.moods)
    assert "cozy" in dict(cozy.moods)
