"""Thumbs feedback: one vote per work, and every vote moves the taste vector."""

from __future__ import annotations

import sqlite3

import pytest

from median import feedback, shelves, taste


def votes(db: sqlite3.Connection, persona_id: int = 1) -> list[tuple[int, str]]:
    rows = db.execute(
        "SELECT work_id, signal FROM rec_feedback WHERE persona_id = ? ORDER BY work_id",
        (persona_id,),
    )
    return [(r["work_id"], r["signal"]) for r in rows]


def test_a_vote_is_stored_and_read_back(db, mini_catalog) -> None:
    assert feedback.get_signal(db, 1, 11) is None
    feedback.set_signal(db, 1, 11, "up", seed_id=1)
    assert feedback.get_signal(db, 1, 11) == "up"
    assert db.execute("SELECT seed_id FROM rec_feedback").fetchone()["seed_id"] == 1


def test_voting_again_replaces_the_vote_instead_of_stacking(db, mini_catalog) -> None:
    feedback.set_signal(db, 1, 11, "up")
    feedback.set_signal(db, 1, 11, "up")
    feedback.set_signal(db, 1, 11, "down")
    assert votes(db) == [(11, "down")]


def test_none_clears_the_vote(db, mini_catalog) -> None:
    feedback.set_signal(db, 1, 11, "up")
    feedback.set_signal(db, 1, 11, None)
    assert votes(db) == []


def test_votes_are_per_persona(db, mini_catalog) -> None:
    feedback.set_signal(db, 1, 11, "up")
    feedback.set_signal(db, 2, 11, "down")
    assert feedback.get_signal(db, 1, 11) == "up"
    assert feedback.get_signal(db, 2, 11) == "down"


def test_a_bad_signal_is_rejected(db, mini_catalog) -> None:
    with pytest.raises(ValueError, match="signal"):
        feedback.set_signal(db, 1, 11, "meh")


def test_votes_move_the_taste_vector(db, mini_catalog) -> None:
    shelves.set_rating(db, 1, 1, 10)
    shelves.set_rating(db, 1, 2, 10)
    assert taste.get_taste_vector(db, 1) is None  # 2 signals: not enough yet
    feedback.set_signal(db, 1, 12, "up")
    assert taste.get_taste_vector(db, 1) is not None
    feedback.set_signal(db, 1, 12, None)
    assert taste.get_taste_vector(db, 1) is None
