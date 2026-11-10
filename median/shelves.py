"""Persona-scoped shelves and ratings.

Every write recomputes the persona's taste vector and commits, so callers (the UI) never have to
remember to (ARCHITECTURE.md section 5: recompute on every rating or shelf change).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from median import taste
from median.browse import row_to_work
from median.models import ShelfItem, Work

SHELVES = ("want", "in_progress", "finished", "favorite")
SHELF_LABELS = {
    "want": "Want",
    "in_progress": "In progress",
    "finished": "Finished",
    "favorite": "Favorites",
}
RATING_RANGE = range(1, 11)  # 10 points = half-star steps on a 5-star display
DEFAULT_RATED_SHELF = "finished"  # rating a work that isn't shelved yet files it here


@dataclass(frozen=True)
class ShelfEntry:
    """A shelved work together with its shelf row."""

    work: Work
    item: ShelfItem


def stars_text(rating: int | None) -> str:
    """Return a 5-star string for a 1-10 rating, using a half star for odd values ('7' -> ★★★½)."""
    if rating is None:
        return "☆☆☆☆☆"
    if rating not in RATING_RANGE:
        raise ValueError(f"rating must be 1-10, got {rating}")
    full, half = divmod(rating, 2)
    return "★" * full + ("½" if half else "") + "☆" * (5 - full - half)


def _check_persona_and_work(conn: sqlite3.Connection, persona_id: int, work_id: int) -> None:
    if conn.execute("SELECT 1 FROM personas WHERE id = ?", (persona_id,)).fetchone() is None:
        raise ValueError(f"unknown persona {persona_id}")
    if conn.execute("SELECT 1 FROM works WHERE id = ?", (work_id,)).fetchone() is None:
        raise ValueError(f"unknown work {work_id}")


def _finish(conn: sqlite3.Connection, persona_id: int) -> None:
    taste.recompute(conn, persona_id)
    conn.commit()


def get_item(conn: sqlite3.Connection, persona_id: int, work_id: int) -> ShelfItem | None:
    """Return the persona's shelf row for this work, or None if it isn't shelved."""
    row = conn.execute(
        "SELECT persona_id, work_id, shelf, rating, note FROM shelf_items "
        "WHERE persona_id = ? AND work_id = ?",
        (persona_id, work_id),
    ).fetchone()
    return ShelfItem(**dict(row)) if row else None


def set_shelf(conn: sqlite3.Connection, persona_id: int, work_id: int, shelf: str) -> None:
    """Put the work on a shelf (moving it if it's elsewhere); an existing rating is kept."""
    if shelf not in SHELVES:
        raise ValueError(f"shelf must be one of {SHELVES}, got {shelf!r}")
    _check_persona_and_work(conn, persona_id, work_id)
    conn.execute(
        "INSERT INTO shelf_items (persona_id, work_id, shelf) VALUES (?, ?, ?) "
        "ON CONFLICT(persona_id, work_id) DO UPDATE SET shelf = excluded.shelf",
        (persona_id, work_id, shelf),
    )
    _finish(conn, persona_id)


def set_rating(conn: sqlite3.Connection, persona_id: int, work_id: int, rating: int | None) -> None:
    """Rate the work 1-10, or clear the rating with None.

    Rating a work that isn't shelved files it under 'finished'. Clearing the rating of a work that
    isn't shelved does nothing. The shelf is never changed on an already-shelved work.
    """
    if rating is not None and rating not in RATING_RANGE:
        raise ValueError(f"rating must be 1-10, got {rating}")
    _check_persona_and_work(conn, persona_id, work_id)
    if get_item(conn, persona_id, work_id) is None:
        if rating is None:
            return
        conn.execute(
            "INSERT INTO shelf_items (persona_id, work_id, shelf, rating) VALUES (?, ?, ?, ?)",
            (persona_id, work_id, DEFAULT_RATED_SHELF, rating),
        )
    else:
        conn.execute(
            "UPDATE shelf_items SET rating = ? WHERE persona_id = ? AND work_id = ?",
            (rating, persona_id, work_id),
        )
    _finish(conn, persona_id)


def remove(conn: sqlite3.Connection, persona_id: int, work_id: int) -> None:
    """Take the work off the persona's shelves, dropping its rating too."""
    conn.execute(
        "DELETE FROM shelf_items WHERE persona_id = ? AND work_id = ?", (persona_id, work_id)
    )
    _finish(conn, persona_id)


def list_shelf(
    conn: sqlite3.Connection, persona_id: int, shelf: str | None = None
) -> list[ShelfEntry]:
    """Return the persona's shelved works, newest first; one shelf, or all when `shelf` is None."""
    if shelf is not None and shelf not in SHELVES:
        raise ValueError(f"shelf must be one of {SHELVES}, got {shelf!r}")
    rows = conn.execute(
        "SELECT w.id, w.media_type, w.title, w.year, w.creators, w.synopsis, w.subjects, "
        "w.popularity, w.cover_url, s.shelf, s.rating, s.note "
        "FROM shelf_items s JOIN works w ON w.id = s.work_id "
        "WHERE s.persona_id = ?" + (" AND s.shelf = ?" if shelf else "") + " "
        "ORDER BY s.added_at DESC, w.id",
        (persona_id, shelf) if shelf else (persona_id,),
    ).fetchall()
    return [
        ShelfEntry(
            row_to_work(r), ShelfItem(persona_id, r["id"], r["shelf"], r["rating"], r["note"])
        )
        for r in rows
    ]


def shelf_counts(conn: sqlite3.Connection, persona_id: int) -> dict[str, int]:
    """Return how many works the persona has on each shelf (zeros included)."""
    counts = dict.fromkeys(SHELVES, 0)
    for row in conn.execute(
        "SELECT shelf, COUNT(*) AS n FROM shelf_items WHERE persona_id = ? GROUP BY shelf",
        (persona_id,),
    ):
        counts[row["shelf"]] = row["n"]
    return counts
