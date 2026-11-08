"""The curated 50-work review set (25 movies + 25 books, 5 known adaptation pairs).

Edit the lists below to change what the team reviews. Pairs are kept together so the review file can
show a film and the book it was adapted from side by side. If a standalone work can't be used (not
found, or no synopsis), the next work from the reserve list takes its place.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from median.db import rebuild_fts
from pipeline.sources import FetchedWork, SkipWork
from pipeline.sources.http import CachedClient
from pipeline.sources.openlibrary import find_book
from pipeline.sources.tmdb import find_movie

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedMovie:
    title: str
    year: int


@dataclass(frozen=True)
class SeedBook:
    title: str
    author: str


@dataclass(frozen=True)
class SeedPair:
    name: str
    movie: SeedMovie
    book: SeedBook


PAIRS = [
    SeedPair(
        "The Shining", SeedMovie("The Shining", 1980), SeedBook("The Shining", "Stephen King")
    ),
    SeedPair("Gone Girl", SeedMovie("Gone Girl", 2014), SeedBook("Gone Girl", "Gillian Flynn")),
    SeedPair(
        "The Fellowship of the Ring",
        SeedMovie("The Lord of the Rings: The Fellowship of the Ring", 2001),
        SeedBook("The Fellowship of the Ring", "J.R.R. Tolkien"),
    ),
    SeedPair(
        "Pride and Prejudice",
        SeedMovie("Pride & Prejudice", 2005),
        SeedBook("Pride and Prejudice", "Jane Austen"),
    ),
    SeedPair("The Martian", SeedMovie("The Martian", 2015), SeedBook("The Martian", "Andy Weir")),
]

MOVIES = [
    SeedMovie("Parasite", 2019),
    SeedMovie("Spirited Away", 2001),
    SeedMovie("Mad Max: Fury Road", 2015),
    SeedMovie("Before Sunrise", 1995),
    SeedMovie("Get Out", 2017),
    SeedMovie("The Grand Budapest Hotel", 2014),
    SeedMovie("Whiplash", 2014),
    SeedMovie("Alien", 1979),
    SeedMovie("Moonlight", 2016),
    SeedMovie("Knives Out", 2019),
    SeedMovie("La La Land", 2016),
    SeedMovie("The Babadook", 2014),
    SeedMovie("Coco", 2017),
    SeedMovie("Lady Bird", 2017),
    SeedMovie("Hereditary", 2018),
    SeedMovie("Amélie", 2001),
    SeedMovie("Her", 2013),
    SeedMovie("Everything Everywhere All at Once", 2022),
    SeedMovie("Toy Story", 1995),
    SeedMovie("Booksmart", 2019),
]
MOVIE_RESERVES = [SeedMovie("Inception", 2010), SeedMovie("The Florida Project", 2017)]

BOOKS = [
    SeedBook("Piranesi", "Susanna Clarke"),
    SeedBook("Circe", "Madeline Miller"),
    SeedBook("Station Eleven", "Emily St. John Mandel"),
    SeedBook("Gilead", "Marilynne Robinson"),
    SeedBook("The Secret History", "Donna Tartt"),
    SeedBook("Kindred", "Octavia E. Butler"),
    SeedBook("Beloved", "Toni Morrison"),
    SeedBook("Normal People", "Sally Rooney"),
    SeedBook("The Name of the Wind", "Patrick Rothfuss"),
    SeedBook("Mexican Gothic", "Silvia Moreno-Garcia"),
    SeedBook("The Left Hand of Darkness", "Ursula K. Le Guin"),
    SeedBook("A Gentleman in Moscow", "Amor Towles"),
    SeedBook("Hyperion", "Dan Simmons"),
    SeedBook("The Night Circus", "Erin Morgenstern"),
    SeedBook("Anxious People", "Fredrik Backman"),
    SeedBook("Charlotte's Web", "E. B. White"),
    SeedBook("The Bear and the Nightingale", "Katherine Arden"),
    SeedBook("Pachinko", "Min Jin Lee"),
    SeedBook("Klara and the Sun", "Kazuo Ishiguro"),
    SeedBook("The Hitchhiker's Guide to the Galaxy", "Douglas Adams"),
]
BOOK_RESERVES = [
    SeedBook("The Midnight Library", "Matt Haig"),
    SeedBook("Babel", "R. F. Kuang"),
    SeedBook("Exit West", "Mohsin Hamid"),
]


@dataclass(frozen=True)
class SampleEntry:
    """One resolved work in the review set."""

    seed_label: str
    pair: str | None  # pair name for adaptation pairs, else None
    work_id: int
    fetched: FetchedWork


@dataclass
class SampleResult:
    entries: list[SampleEntry] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def upsert_work(conn: sqlite3.Connection, work: FetchedWork) -> int:
    """Insert or refresh a work (keyed on its external id) and return its `works.id`."""
    row = conn.execute(
        "SELECT work_id FROM external_ids WHERE source = ? AND external_id = ?",
        (work.source, work.external_id),
    ).fetchone()
    values = (
        work.media_type,
        work.title,
        work.year,
        json.dumps(work.creators),
        work.synopsis,
        json.dumps(work.subjects),
        work.cover_url,
        work.popularity_raw,
    )
    if row:
        conn.execute(
            "UPDATE works SET media_type = ?, title = ?, year = ?, creators = ?, synopsis = ?, "
            "subjects = ?, cover_url = ?, popularity_raw = ? WHERE id = ?",
            (*values, row["work_id"]),
        )
        return row["work_id"]
    cur = conn.execute(
        "INSERT INTO works (media_type, title, year, creators, synopsis, subjects, cover_url, "
        "popularity_raw) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        values,
    )
    conn.execute(
        "INSERT INTO external_ids (work_id, source, external_id) VALUES (?, ?, ?)",
        (cur.lastrowid, work.source, work.external_id),
    )
    return cur.lastrowid


def _describe(seed_label: str, work: FetchedWork) -> str:
    flag = f"  ⚠ {work.match_note}" if work.match_note else ""
    creators = ", ".join(work.creators) or "unknown"
    return (
        f"{seed_label:<48} -> {work.title} ({work.year}) by {creators} "
        f"[{work.source}:{work.external_id}]{flag}"
    )


def load_sample(
    conn: sqlite3.Connection, tmdb: CachedClient, openlibrary: CachedClient
) -> SampleResult:
    """Fetch the review set from TMDB and Open Library and upsert it into the DB.

    Logs a resolution report (seed -> matched work) so wrong matches are visible before any LLM
    spend. Responses are cached, so a rerun makes no network calls.
    """
    result = SampleResult()

    def add(label: str, pair: str | None, work: FetchedWork) -> None:
        work_id = upsert_work(conn, work)
        result.entries.append(SampleEntry(label, pair, work_id, work))
        logger.info(_describe(label, work))

    def fetch_movie(seed: SeedMovie) -> FetchedWork:
        return find_movie(tmdb, seed.title, seed.year)

    def fetch_book(seed: SeedBook) -> FetchedWork:
        return find_book(openlibrary, seed.title, seed.author)

    for pair in PAIRS:
        for label, fetch, seed in (
            (f"{pair.movie.title} ({pair.movie.year})", fetch_movie, pair.movie),
            (f"{pair.book.title} / {pair.book.author}", fetch_book, pair.book),
        ):
            try:
                add(label, pair.name, fetch(seed))
            except SkipWork as exc:
                result.failures.append(f"{label}: {exc}")
                logger.warning("SKIPPED %s: %s", label, exc)

    for seeds, reserves, fetch, labeler in (
        (MOVIES, list(MOVIE_RESERVES), fetch_movie, lambda s: f"{s.title} ({s.year})"),
        (BOOKS, list(BOOK_RESERVES), fetch_book, lambda s: f"{s.title} / {s.author}"),
    ):
        for seed in seeds:
            candidate = seed
            while True:
                try:
                    add(labeler(candidate), None, fetch(candidate))
                    break
                except SkipWork as exc:
                    result.failures.append(f"{labeler(candidate)}: {exc}")
                    logger.warning("SKIPPED %s: %s", labeler(candidate), exc)
                    if not reserves:
                        break
                    candidate = reserves.pop(0)  # each reserve is used at most once

    conn.commit()
    rebuild_fts(conn)
    conn.commit()
    return result


def forced_from_entries(
    entries: list[SampleEntry],
) -> tuple[list[int], list[str], list[tuple[int, str]]]:
    """Return (tmdb movie ids, OL work ids, adaptation pairs) for the review set's entries."""
    movies = [int(e.fetched.external_id) for e in entries if e.fetched.media_type == "movie"]
    books = [e.fetched.external_id for e in entries if e.fetched.media_type == "book"]
    by_pair: dict[str, dict[str, str]] = {}
    for entry in entries:
        if entry.pair:
            by_pair.setdefault(entry.pair, {})[entry.fetched.media_type] = entry.fetched.external_id
    pairs = [
        (int(p["movie"]), p["book"]) for p in by_pair.values() if {"movie", "book"} <= p.keys()
    ]
    return movies, books, pairs
