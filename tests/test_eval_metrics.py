"""The eval metrics, checked on the fixture catalog where the right answers are known."""

from __future__ import annotations

import sqlite3

import pytest

from tests.eval import metrics

# fixture ids: movie -> book adaptations 1->11, 2->12, 4->14, 5->15, 9->19, 8->18 (unverified link)
PAIRS = [(11, 1), (12, 2), (14, 4), (15, 5), (19, 9), (18, 8)]  # (book_id, movie_id)


def test_every_fixture_adaptation_is_found_from_the_book_side(
    db: sqlite3.Connection, mini_catalog: dict
) -> None:
    recall = metrics.adaptation_recall(db, PAIRS)
    assert (recall.hits, recall.total, recall.value) == (6, 6, 1.0)
    assert recall.misses == []


def test_and_from_the_film_side(db: sqlite3.Connection, mini_catalog: dict) -> None:
    assert metrics.adaptation_recall(db, PAIRS, reverse=True).value == 1.0


def test_a_wrong_pair_is_reported_as_a_miss_by_name(
    db: sqlite3.Connection, mini_catalog: dict
) -> None:
    # the fixture has 10 films, so only a small k can miss; film 10 is unrelated to book 11
    recall = metrics.adaptation_recall(db, [(11, 1), (11, 10)], k=3)
    assert recall.value == 0.5
    assert recall.misses == ["The Shining -> Mad Max: Fury Road"]


def test_the_cross_cluster_adaptation_is_found_through_its_link(
    db: sqlite3.Connection, mini_catalog: dict
) -> None:
    """Jurassic Park's film and novel are in different vibe clusters; only the link joins them."""
    assert metrics.adaptation_recall(db, [(19, 9)], reverse=False).value == 1.0


def test_the_unlinked_hobbit_pair_is_found_by_vibe_alone(
    db: sqlite3.Connection, mini_catalog: dict
) -> None:
    assert metrics.vibe_recall(db, [(6, 16)]).value == 1.0  # Hobbit film -> Hobbit novel


def test_a_far_apart_vibe_pair_is_a_miss(db: sqlite3.Connection, mini_catalog: dict) -> None:
    recall = metrics.vibe_recall(db, [(1, 20)], k=2)  # a haunted film vs a sci-fi techno-thriller
    assert recall.value == 0.0 and len(recall.misses) == 1


def test_medium_mixing_is_not_near_zero_when_vibes_cross_media(
    db: sqlite3.Connection, mini_catalog: dict
) -> None:
    share = metrics.medium_mixing(db, [1, 11, 3, 14])
    assert 0.3 < share < 0.9


def test_recall_value_of_an_empty_set_is_zero() -> None:
    assert metrics.Recall().value == 0.0


def test_refs_resolve_to_database_ids(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT INTO works (id, media_type, title) VALUES (7, 'movie', 'X'), (8, 'book', 'Y')"
    )
    db.execute("INSERT INTO external_ids VALUES (7, 'tmdb', '694'), (8, 'openlibrary', 'OL1W')")
    assert metrics.ref_id(db, "tmdb:694") == 7 and metrics.ref_id(db, "ol:OL1W") == 8
    assert metrics.ref_id(db, "tmdb:999") is None


@pytest.mark.parametrize("bad", ["694", "x:1:2"])
def test_malformed_refs_are_rejected(db: sqlite3.Connection, bad: str) -> None:
    with pytest.raises(ValueError):
        metrics.ref_id(db, bad)
