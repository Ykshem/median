"""Find candidate film->book adaptations that Wikidata doesn't know about (it covers only ~1/3).

Two generators, both writing `adaptation_of` links with `verified=0` for `link_check` to judge:
  tmdb_credits : the film's TMDB crew credits a "Novel"/"Book"/"Author", and a catalog book is by
                 that person (fuzzy name match) with a similar title, or is their only book here.
  title_match  : no such credit, but a catalog book has exactly the film's title and is not newer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from pipeline.sources import normalize_title
from pipeline.sources.http import CachedClient
from pipeline.sources.tmdb import crew_names, raw_movie

logger = logging.getLogger(__name__)

NOVEL_JOBS = {"Novel", "Book", "Author"}
AUTHOR_SCORE = 95  # rapidfuzz token_sort_ratio on person names ("Peter Brown" != "Peter C. Brown")
TITLE_SCORE = 85  # see title_score(); sequels like 'Freed'/'Darker' score ~81
CONTAINED_TITLE_SCORE = 90  # one title wholly inside the other, e.g. a film's subtitle
CREDIT_BASE = 0.6
TITLE_ONLY_CONFIDENCE = 0.4


@dataclass(frozen=True)
class Candidate:
    movie_id: int
    book_id: int
    source: str  # 'tmdb_credits' | 'title_match'
    confidence: float


@dataclass(frozen=True)
class _Work:
    id: int
    title: str  # normalised
    year: int | None
    creators: tuple[str, ...]  # normalised
    external_id: str


def _works(conn: sqlite3.Connection, media_type: str, source: str) -> list[_Work]:
    rows = conn.execute(
        "SELECT w.id, w.title, w.year, w.creators, e.external_id FROM works w "
        "JOIN external_ids e ON e.work_id = w.id AND e.source = ? WHERE w.media_type = ?",
        (source, media_type),
    ).fetchall()
    return [
        _Work(
            r["id"],
            normalize_title(r["title"]),
            r["year"],
            tuple(normalize_title(c) for c in json.loads(r["creators"] or "[]")),
            r["external_id"],
        )
        for r in rows
    ]


def title_score(film: str, book: str) -> float:
    """Score how alike two normalised titles are, 0-100.

    Identical is 100; one title appearing whole inside the other ("the hunger games" in "the hunger
    games mockingjay part 1") is 90; otherwise the plain edit-distance ratio. Unlike WRatio this
    does not rate unrelated short titles as similar.
    """
    if film == book:
        return 100.0
    if f" {book} " in f" {film} " or f" {film} " in f" {book} ":
        return CONTAINED_TITLE_SCORE
    return float(fuzz.ratio(film, book))


def _not_newer(film: _Work, book: _Work) -> bool:
    """A book can only be adapted by a film released in the same year or later (1 year of slack)."""
    return film.year is None or book.year is None or book.year <= film.year + 1


def find_candidates(conn: sqlite3.Connection, tmdb: CachedClient) -> list[Candidate]:
    """Return candidate adaptation links for every catalog movie, skipping already-linked pairs."""
    movies = _works(conn, "movie", "tmdb")
    books = _works(conn, "book", "openlibrary")
    by_author: dict[str, list[_Work]] = {}
    by_title: dict[str, list[_Work]] = {}
    for book in books:
        by_title.setdefault(book.title, []).append(book)
        for creator in book.creators:
            by_author.setdefault(creator, []).append(book)
    author_names = list(by_author)
    linked = {
        (r["from_work"], r["to_work"])
        for r in conn.execute("SELECT from_work, to_work FROM work_links")
    }

    candidates: list[Candidate] = []
    for film in movies:
        credited = crew_names(raw_movie(tmdb, film.external_id), NOVEL_JOBS)
        found: dict[int, float] = {}
        for name in credited:
            matches = process.extract(
                normalize_title(name),
                author_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=AUTHOR_SCORE,
                limit=5,
            )
            for author, _score, _index in matches:
                shelf = by_author[author]
                for book in shelf:
                    if not _not_newer(film, book):
                        continue
                    score = title_score(film.title, book.title)
                    if score >= TITLE_SCORE:
                        confidence = CREDIT_BASE + 0.3 * score / 100
                    elif len(shelf) == 1:
                        confidence = CREDIT_BASE
                    else:
                        continue
                    found[book.id] = max(found.get(book.id, 0.0), round(confidence, 2))
        source = "tmdb_credits"
        if not found:
            source = "title_match"
            for book in by_title.get(film.title, []):
                if book.year is not None and _not_newer(film, book):
                    found[book.id] = TITLE_ONLY_CONFIDENCE
        candidates.extend(
            Candidate(film.id, book_id, source, confidence)
            for book_id, confidence in found.items()
            if (film.id, book_id) not in linked
        )
    return candidates


def match_links(conn: sqlite3.Connection, tmdb: CachedClient) -> dict[str, int]:
    """Insert candidate links (`verified=0`) and return how many were added per source."""
    added = {"tmdb_credits": 0, "title_match": 0}
    for cand in find_candidates(conn, tmdb):
        cursor = conn.execute(
            "INSERT OR IGNORE INTO work_links "
            "(from_work, to_work, link_type, source, confidence, verified) "
            "VALUES (?, ?, 'adaptation_of', ?, ?, 0)",
            (cand.movie_id, cand.book_id, cand.source, cand.confidence),
        )
        added[cand.source] += cursor.rowcount
    conn.commit()
    logger.info("match: added %s unverified candidate link(s)", added)
    return added
