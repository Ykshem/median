"""Write fetched works and Wikidata links into the database (the DB side of the fetch steps)."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.sample_plan import bare_id
from pipeline.sample_seed import upsert_work
from pipeline.sources import SkipWork
from pipeline.sources.http import CachedClient
from pipeline.sources.ol_dumps import RecordCache, has_description, to_fetched_work
from pipeline.sources.tmdb import get_movie

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    loaded: int = 0
    skipped: list[str] = field(default_factory=list)


def load_tmdb_movies(conn: sqlite3.Connection, client: CachedClient, ids: list[int]) -> LoadReport:
    """Fetch each TMDB movie (cached) and upsert it. Movies with no overview are skipped."""
    report = LoadReport()
    for n, movie_id in enumerate(ids, 1):
        try:
            upsert_work(conn, get_movie(client, movie_id))
            report.loaded += 1
        except SkipWork as exc:
            report.skipped.append(f"tmdb:{movie_id}: {exc}")
        if n % 100 == 0:
            conn.commit()
            logger.info("  TMDB %d / %d", n, len(ids))
    conn.commit()
    return report


def load_openlibrary_books(
    conn: sqlite3.Connection,
    ol_ids: list[str],
    *,
    reading_counts: Mapping[str, int],
    works_cache: RecordCache,
    authors_cache: RecordCache,
    works_dump: Path,
    authors_dump: Path,
    min_description_chars: int,
) -> LoadReport:
    """Upsert the given Open Library works, reading the dumps only for keys not seen before.

    Works without a description of at least `min_description_chars` are skipped.
    """
    report = LoadReport()
    keys = [f"/works/{bare_id(i)}" for i in ol_ids]
    projected = works_cache.get_many(keys, works_dump)
    usable = {k: p for k, p in projected.items() if has_description(p, min_description_chars)}
    author_keys = sorted({a for p in usable.values() for a in p["author_keys"]})
    authors = authors_cache.get_many(author_keys, authors_dump)
    names = {key: (record or {}).get("name", "") for key, record in authors.items()}
    counts = {bare_id(k): n for k, n in reading_counts.items()}
    for key in keys:
        if key not in usable:
            report.skipped.append(f"{bare_id(key)}: missing from the dump or no description")
            continue
        popularity = float(counts.get(bare_id(key), 0))  # unshelved books count as 0
        upsert_work(conn, to_fetched_work(key, usable[key], names, popularity))
        report.loaded += 1
    conn.commit()
    return report


def _external_ids(conn: sqlite3.Connection, source: str) -> dict[str, int]:
    rows = conn.execute("SELECT external_id, work_id FROM external_ids WHERE source = ?", (source,))
    return {r["external_id"]: r["work_id"] for r in rows}


def link_wikidata(
    conn: sqlite3.Connection,
    adaptations: list[tuple[int, str]],
    series: list[tuple[int, str, str]],
) -> dict[str, int]:
    """Insert Wikidata links whose two ends are both in the catalog; return counts by link type.

    Wikidata links are trusted (`verified=1`) and replace an unverified candidate for the same pair.
    """
    movies = _external_ids(conn, "tmdb")
    books = _external_ids(conn, "openlibrary")
    sql = (
        "INSERT INTO work_links (from_work, to_work, link_type, source, confidence, verified) "
        "VALUES (?, ?, ?, 'wikidata', 1.0, 1) "
        "ON CONFLICT (from_work, to_work, link_type) DO UPDATE SET "
        "source = 'wikidata', confidence = 1.0, verified = 1"
    )
    counts = {"adaptation_of": 0, "same_franchise": 0}
    pairs = [(t, o, "adaptation_of") for t, o in adaptations]
    pairs += [(t, o, "same_franchise") for t, o, _ in series]
    for tmdb_id, ol_id, link_type in dict.fromkeys(pairs):
        movie, book = movies.get(str(tmdb_id)), books.get(ol_id)
        if movie and book:
            conn.execute(sql, (movie, book, link_type))
            counts[link_type] += 1
    conn.commit()
    return counts
