from __future__ import annotations

import math
import sqlite3

import pytest

from median import config, taste
from median.db import vec_to_blob


def vec(**dims: float) -> list[float]:
    """A 1536-dim vector with the named dimensions set (a=0, b=1, c=2, d=3), the rest zero."""
    out = [0.0] * config.EMBED_DIMS
    for name, value in dims.items():
        out["abcdefgh".index(name)] = value
    return out


def add_work(
    db: sqlite3.Connection, work_id: int, vector: list[float] | None, media="movie"
) -> None:
    db.execute(
        "INSERT INTO works (id, media_type, title) VALUES (?, ?, ?)",
        (work_id, media, f"Work {work_id}"),
    )
    if vector is not None:
        db.execute(
            "INSERT INTO vec_vibes (work_id, embedding, media_type) VALUES (?, ?, ?)",
            (work_id, vec_to_blob(vector), media),
        )


def shelve(db, persona_id, work_id, shelf="finished", rating=None) -> None:
    db.execute(
        "INSERT INTO shelf_items (persona_id, work_id, shelf, rating) VALUES (?, ?, ?, ?)",
        (persona_id, work_id, shelf, rating),
    )


# math --------------------------------------------------------------------------------------------


def test_rating_weights_run_from_minus_one_to_plus_one() -> None:
    assert taste.rating_weight(10) == pytest.approx(1.0)
    assert taste.rating_weight(1) == pytest.approx(-1.0)
    assert taste.rating_weight(5) == pytest.approx(-0.5 / 4.5)
    assert taste.rating_weight(6) == pytest.approx(0.5 / 4.5)


def test_weighted_direction_is_a_unit_vector_toward_the_heavier_input() -> None:
    result = taste.weighted_direction([(vec(a=1), 3.0), (vec(b=1), 1.0)])
    assert math.sqrt(sum(x * x for x in result)) == pytest.approx(1.0)
    assert result[0] == pytest.approx(3 / math.sqrt(10))
    assert result[1] == pytest.approx(1 / math.sqrt(10))


def test_a_negative_weight_pushes_away() -> None:
    result = taste.weighted_direction([(vec(a=1), 1.0), (vec(b=1), -1.0)])
    assert result[0] > 0 > result[1]


def test_weights_that_cancel_out_give_no_direction() -> None:
    assert taste.weighted_direction([(vec(a=1), 1.0), (vec(a=1), -1.0)]) is None
    assert taste.weighted_direction([]) is None


def test_cosine_and_normalize_handle_zero_vectors() -> None:
    assert taste.cosine(vec(a=1), vec(a=2)) == pytest.approx(1.0)
    assert taste.cosine(vec(a=1), vec(b=1)) == pytest.approx(0.0)
    assert taste.cosine(vec(a=1), vec()) == 0.0
    assert taste.normalize(vec()) is None


# signals -----------------------------------------------------------------------------------------


def test_ratings_favorites_and_feedback_become_weights(db: sqlite3.Connection) -> None:
    for i in range(1, 6):
        add_work(db, i, vec(a=1))
    shelve(db, 1, 1, rating=10)
    shelve(db, 1, 2, shelf="favorite")  # no rating: +1.0
    shelve(db, 1, 3, rating=2)
    shelve(db, 1, 4, shelf="want")  # no rating, not a favorite: no signal
    db.execute("INSERT INTO rec_feedback (persona_id, work_id, signal) VALUES (1, 5, 'up')")
    db.execute("INSERT INTO rec_feedback (persona_id, work_id, signal) VALUES (1, 5, 'down')")
    db.execute("INSERT INTO rec_feedback (persona_id, work_id, signal) VALUES (1, 4, 'down')")
    signals = taste.collect_signals(db, 1)
    assert signals.weights[1] == pytest.approx(1.0)
    assert signals.weights[2] == pytest.approx(1.0)
    assert signals.weights[3] == pytest.approx(taste.rating_weight(2))
    assert signals.weights[4] == pytest.approx(-0.5)
    assert signals.weights[5] == pytest.approx(0.0)  # +0.5 and -0.5
    assert signals.count == 6  # 3 shelf signals + 3 feedback rows
    assert signals.liked() == [1, 2]


def test_a_rated_favorite_uses_the_larger_weight_not_both(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1))
    add_work(db, 2, vec(a=1))
    shelve(db, 1, 1, shelf="favorite", rating=10)  # max(1.0, 1.0)
    shelve(db, 1, 2, shelf="favorite", rating=3)  # the favorite weight beats a low rating
    weights = taste.collect_signals(db, 1).weights
    assert weights[1] == pytest.approx(1.0) and weights[2] == pytest.approx(1.0)


def test_works_without_a_vector_are_ignored(db: sqlite3.Connection) -> None:
    add_work(db, 1, None)
    add_work(db, 2, vec(a=1))
    shelve(db, 1, 1, rating=10)
    shelve(db, 1, 2, rating=10)
    signals = taste.collect_signals(db, 1)
    assert set(signals.weights) == {2} and signals.count == 1


def test_signals_are_per_persona(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1))
    shelve(db, 1, 1, rating=10)
    assert taste.collect_signals(db, 2).count == 0


# taste vector ------------------------------------------------------------------------------------


def _seed_likes(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1))
    add_work(db, 2, vec(a=1, b=0.3))
    add_work(db, 3, vec(b=1))
    add_work(db, 4, vec(c=1))  # disliked
    for work_id, rating in [(1, 10), (2, 9), (3, 8), (4, 1)]:
        shelve(db, 1, work_id, rating=rating)


def test_the_vector_stays_null_until_there_are_three_signals(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1))
    add_work(db, 2, vec(a=1))
    shelve(db, 1, 1, rating=10)
    shelve(db, 1, 2, rating=10)
    assert taste.compute_taste_vector(db, 1) is None
    assert taste.recompute(db, 1) is False
    assert db.execute("SELECT taste_vector FROM personas WHERE id = 1").fetchone()[0] is None


def test_the_vector_points_toward_liked_works_and_away_from_disliked(
    db: sqlite3.Connection,
) -> None:
    _seed_likes(db)
    result = taste.compute_taste_vector(db, 1)
    assert math.sqrt(sum(x * x for x in result)) == pytest.approx(1.0)
    assert taste.cosine(result, vec(a=1)) > 0.5
    assert taste.cosine(result, vec(c=1)) < 0  # pushed away from the 1-star work


def test_recompute_stores_a_blob_and_a_timestamp(db: sqlite3.Connection) -> None:
    _seed_likes(db)
    assert taste.recompute(db, 1) is True
    row = db.execute("SELECT taste_vector, updated_at FROM personas WHERE id = 1").fetchone()
    assert row["updated_at"].startswith("20")
    stored = taste.get_taste_vector(db, 1)
    assert stored == pytest.approx(taste.compute_taste_vector(db, 1), abs=1e-6)
    assert len(row["taste_vector"]) == config.EMBED_DIMS * 4


def test_recompute_follows_changes_and_can_clear_the_vector(db: sqlite3.Connection) -> None:
    _seed_likes(db)
    taste.recompute(db, 1)
    before = taste.get_taste_vector(db, 1)
    db.execute("UPDATE shelf_items SET rating = 10 WHERE work_id = 4")  # now loves the odd one out
    taste.recompute(db, 1)
    assert taste.cosine(taste.get_taste_vector(db, 1), vec(c=1)) > taste.cosine(before, vec(c=1))
    db.execute("DELETE FROM shelf_items WHERE persona_id = 1")
    assert taste.recompute(db, 1) is False
    assert taste.get_taste_vector(db, 1) is None


def test_thumbs_down_moves_the_vector_away(db: sqlite3.Connection) -> None:
    _seed_likes(db)
    add_work(db, 9, vec(d=1))
    taste.recompute(db, 1)
    base = taste.cosine(taste.get_taste_vector(db, 1), vec(d=1))
    db.execute("INSERT INTO rec_feedback (persona_id, work_id, signal) VALUES (1, 9, 'down')")
    taste.recompute(db, 1)
    assert taste.cosine(taste.get_taste_vector(db, 1), vec(d=1)) < base


def test_recompute_all_counts_personas_that_have_a_vector(db: sqlite3.Connection) -> None:
    _seed_likes(db)  # persona 1 only
    assert taste.recompute_all(db) == 1
    assert taste.get_taste_vector(db, 2) is None
