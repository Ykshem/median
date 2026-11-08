"""Recommendation quality metrics (ARCHITECTURE.md section 10), as plain functions.

Kept separate from the eval tests so they can be checked against the small fixture catalog, whose
right answers are known, before being pointed at the real one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from median import recommend

K = 10


@dataclass
class Recall:
    hits: int = 0
    total: int = 0
    misses: list[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def record(self, hit: bool, label: str) -> None:
        self.total += 1
        if hit:
            self.hits += 1
        else:
            self.misses.append(label)


def work_id(conn: sqlite3.Connection, source: str, external_id: str) -> int | None:
    """Return the database id of a work from its source and external id, or None."""
    row = conn.execute(
        "SELECT work_id FROM external_ids WHERE source = ? AND external_id = ?",
        (source, external_id),
    ).fetchone()
    return row["work_id"] if row else None


def ref_id(conn: sqlite3.Connection, ref: str) -> int | None:
    """Resolve a gold reference like 'tmdb:694' or 'ol:OL81633W' to a database id."""
    kind, ident = ref.split(":")
    return work_id(conn, "tmdb" if kind == "tmdb" else "openlibrary", ident)


def media_of(conn: sqlite3.Connection, wid: int) -> str:
    return conn.execute("SELECT media_type FROM works WHERE id = ?", (wid,)).fetchone()[0]


def title_of(conn: sqlite3.Connection, wid: int) -> str:
    return conn.execute("SELECT title FROM works WHERE id = ?", (wid,)).fetchone()[0]


def adaptation_recall(
    conn: sqlite3.Connection, pairs: list[tuple[int, int]], *, reverse: bool = False, k: int = K
) -> Recall:
    """Share of (book_id, movie_id) pairs where the film is in overlap(book, 'movie')[:K].

    With reverse=True, the book is looked for in overlap(film, 'book')[:K] instead.
    """
    recall = Recall()
    for book_id, movie_id in pairs:
        seed, target, medium = (
            (movie_id, book_id, "book") if reverse else (book_id, movie_id, "movie")
        )
        found = [r.work_id for r in recommend.overlap(conn, seed, medium, k=k)]
        recall.record(target in found, f"{title_of(conn, seed)} -> {title_of(conn, target)}")
    return recall


def vibe_recall(conn: sqlite3.Connection, pairs: list[tuple[int, int]], *, k: int = K) -> Recall:
    """Share of (seed_id, target_id) pairs whose target is in overlap(seed, <its medium>)[:k]."""
    recall = Recall()
    for seed_id, target_id in pairs:
        found = [
            r.work_id for r in recommend.overlap(conn, seed_id, media_of(conn, target_id), k=k)
        ]
        recall.record(
            target_id in found, f"{title_of(conn, seed_id)} -> {title_of(conn, target_id)}"
        )
    return recall


def medium_mixing(conn: sqlite3.Connection, seeds: list[int], *, k: int = K) -> float:
    """Share of overlap(seed, 'any')[:k] results that are in the other medium than the seed.

    Near 0 would mean the vibe cards leak the medium and works only match their own kind.
    """
    other = total = 0
    for seed_id in seeds:
        seed_media = media_of(conn, seed_id)
        for rec in recommend.overlap(conn, seed_id, "any", k=k):
            total += 1
            other += media_of(conn, rec.work_id) != seed_media
    return other / total if total else 0.0
