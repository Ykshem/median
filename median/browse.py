"""Catalog browsing: search and filter works, and the data behind the Work page.

Pure read-only helpers over the catalog. The UI calls these; it never writes SQL. All SQL here is
parameterized, and user text only ever reaches FTS5 through `fts_query`, which quotes every token.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

from median import recommend
from median.db import vec_to_blob
from median.models import Link, Recommendation, VibeCard, Work

SORTS = ("relevance", "popular", "year_newest", "year_oldest", "title")
MEDIA_FILTERS = ("any", "movie", "book")
_OTHER_MEDIUM = {"movie": "book", "book": "movie"}
_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SearchFilters:
    """What the Browse page asks for. `query` matches titles and creators; `mood` is one word."""

    query: str = ""
    media_type: str = "any"
    year_min: int | None = None
    year_max: int | None = None
    mood: str | None = None
    sort: str = "relevance"


@dataclass(frozen=True)
class LinkedWork:
    """A work connected to the one being viewed by a verified link."""

    work: Work
    link_type: str
    relation: str  # direction-aware label, e.g. "Adapted from" or "Same series"
    confidence: float


@dataclass(frozen=True)
class SimilarWork:
    """A vibe-similar work in the other medium, with the reasons the scorer gave."""

    work: Work
    recommendation: Recommendation


@dataclass(frozen=True)
class WorkDetail:
    """Everything the Work page shows for one work."""

    work: Work
    card: VibeCard | None
    linked: list[LinkedWork] = field(default_factory=list)
    similar: list[SimilarWork] = field(default_factory=list)
    has_vector: bool = False


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 query: every word quoted, the last one a prefix match.

    Returns None when the text has no searchable words. Quoting stops FTS5 syntax (AND, NEAR, `*`,
    quotes, colons) in user input from being interpreted or raising an error.
    """
    tokens = _TOKEN.findall(text)
    if not tokens:
        return None
    parts = [f'"{t}"' for t in tokens]
    parts[-1] += "*"
    return " ".join(parts)


def row_to_work(row: sqlite3.Row) -> Work:
    return Work(
        id=row["id"],
        media_type=row["media_type"],
        title=row["title"],
        year=row["year"],
        creators=json.loads(row["creators"]) if row["creators"] else [],
        synopsis=row["synopsis"],
        subjects=json.loads(row["subjects"]) if row["subjects"] else [],
        popularity=row["popularity"],
        cover_url=row["cover_url"],
    )


_WORK_COLUMNS = (
    "w.id, w.media_type, w.title, w.year, w.creators, w.synopsis, w.subjects, "
    "w.popularity, w.cover_url"
)


def get_work(conn: sqlite3.Connection, work_id: int) -> Work | None:
    """Return the work with this id, or None."""
    row = conn.execute(f"SELECT {_WORK_COLUMNS} FROM works w WHERE w.id = ?", (work_id,)).fetchone()  # noqa: S608
    return row_to_work(row) if row else None


def works_by_ids(conn: sqlite3.Connection, work_ids: list[int]) -> dict[int, Work]:
    """Return {id: Work} for the ids that exist."""
    if not work_ids:
        return {}
    marks = ",".join("?" * len(work_ids))
    rows = conn.execute(
        f"SELECT {_WORK_COLUMNS} FROM works w WHERE w.id IN ({marks})",  # noqa: S608
        work_ids,
    )
    return {r["id"]: row_to_work(r) for r in rows}


def get_card(conn: sqlite3.Connection, work_id: int) -> VibeCard | None:
    """Return the work's validated vibe card, or None if it has none (or it no longer validates)."""
    row = conn.execute("SELECT card_json FROM vibe_cards WHERE work_id = ?", (work_id,)).fetchone()
    if row is None:
        return None
    try:
        return VibeCard.model_validate_json(row["card_json"])
    except ValueError:
        return None


def has_vector(conn: sqlite3.Connection, work_id: int) -> bool:
    """Return True if the work has a vibe embedding in the index."""
    row = conn.execute("SELECT 1 FROM vec_vibes WHERE work_id = ?", (work_id,)).fetchone()
    return row is not None


def count_vectors(conn: sqlite3.Connection) -> int:
    """Return how many works have a vibe embedding (0 until the embed step has run)."""
    return conn.execute("SELECT COUNT(*) AS n FROM vec_vibes").fetchone()["n"]


# Search ---------------------------------------------------------------------------------------


def _where(filters: SearchFilters) -> tuple[str, str, list[object]]:
    """Return (join, where, params) for the filters. Raises ValueError on bad input."""
    if filters.media_type not in MEDIA_FILTERS:
        raise ValueError(f"media_type must be one of {MEDIA_FILTERS}")
    joins: list[str] = []
    wheres: list[str] = []
    params: list[object] = []
    match = fts_query(filters.query)
    if match is not None:
        joins.append("JOIN works_fts f ON f.rowid = w.id")
        wheres.append("works_fts MATCH ?")
        params.append(match)
    if filters.media_type != "any":
        wheres.append("w.media_type = ?")
        params.append(filters.media_type)
    if filters.year_min is not None:
        wheres.append("w.year >= ?")
        params.append(filters.year_min)
    if filters.year_max is not None:
        wheres.append("w.year <= ?")
        params.append(filters.year_max)
    if filters.mood:
        wheres.append(
            "EXISTS (SELECT 1 FROM vibe_cards c, json_each(c.card_json, '$.mood') m "
            "WHERE c.work_id = w.id AND lower(m.value) = ?)"
        )
        params.append(filters.mood.lower())
    return " ".join(joins), (" WHERE " + " AND ".join(wheres)) if wheres else "", params


def _order(filters: SearchFilters, *, searching: bool) -> str:
    if filters.sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}")
    if filters.sort == "relevance":
        # Without a query there is nothing to rank by, so fall back to popularity.
        return "ORDER BY bm25(works_fts), w.id" if searching else "ORDER BY w.popularity DESC, w.id"
    return {
        "popular": "ORDER BY w.popularity DESC, w.id",
        "year_newest": "ORDER BY w.year IS NULL, w.year DESC, w.id",
        "year_oldest": "ORDER BY w.year IS NULL, w.year ASC, w.id",
        "title": "ORDER BY w.title COLLATE NOCASE, w.id",
    }[filters.sort]


def search_works(
    conn: sqlite3.Connection, filters: SearchFilters, limit: int = 24, offset: int = 0
) -> list[Work]:
    """Return one page of works matching the filters, in the requested order."""
    if limit < 1 or offset < 0:
        raise ValueError("limit must be >= 1 and offset >= 0")
    joins, where, params = _where(filters)
    order = _order(filters, searching=fts_query(filters.query) is not None)
    rows = conn.execute(
        f"SELECT {_WORK_COLUMNS} FROM works w {joins}{where} {order} LIMIT ? OFFSET ?",  # noqa: S608
        [*params, limit, offset],
    )
    return [row_to_work(r) for r in rows]


def count_matches(conn: sqlite3.Connection, filters: SearchFilters) -> int:
    """Return how many works match the filters (for paging)."""
    joins, where, params = _where(filters)
    row = conn.execute(f"SELECT COUNT(*) AS n FROM works w {joins}{where}", params).fetchone()  # noqa: S608
    return row["n"]


def list_moods(conn: sqlite3.Connection, limit: int = 40) -> list[str]:
    """Return the most common mood words across all vibe cards (lower-case), most common first."""
    rows = conn.execute(
        "SELECT lower(m.value) AS mood, COUNT(*) AS n FROM vibe_cards c, "
        "json_each(c.card_json, '$.mood') m GROUP BY mood ORDER BY n DESC, mood LIMIT ?",
        (limit,),
    )
    return [r["mood"] for r in rows]


def list_themes(conn: sqlite3.Connection, limit: int = 40) -> list[str]:
    """Return the most common theme words across all vibe cards (lower-case), most common first."""
    rows = conn.execute(
        "SELECT lower(t.value) AS theme, COUNT(*) AS n FROM vibe_cards c, "
        "json_each(c.card_json, '$.themes') t GROUP BY theme ORDER BY n DESC, theme LIMIT ?",
        (limit,),
    )
    return [r["theme"] for r in rows]


def semantic_search(
    conn: sqlite3.Connection, vector: list[float], media_type: str = "any", limit: int = 10
) -> list[int]:
    """Return ids of the works whose vibe vectors are nearest `vector`, best first."""
    if media_type not in MEDIA_FILTERS:
        raise ValueError(f"media_type must be one of {MEDIA_FILTERS}")
    sql = "SELECT work_id FROM vec_vibes WHERE embedding MATCH ? AND k = ?"
    params: list[object] = [vec_to_blob(vector), limit]
    if media_type != "any":
        sql += " AND media_type = ?"
        params.append(media_type)
    return [r["work_id"] for r in conn.execute(sql + " ORDER BY distance", params)]


def year_bounds(conn: sqlite3.Connection) -> tuple[int, int] | None:
    """Return (earliest, latest) year in the catalog, or None if no work has a year."""
    row = conn.execute("SELECT MIN(year) AS lo, MAX(year) AS hi FROM works").fetchone()
    return (row["lo"], row["hi"]) if row["lo"] is not None else None


# Work detail ----------------------------------------------------------------------------------


def _relation(link: Link, work_id: int) -> str:
    """Return a direction-aware label for how `work_id` relates to the other end of `link`."""
    if link.link_type == "adaptation_of":
        # from_work (the film) is adapted from to_work (the book)
        return "Adapted from" if link.from_work == work_id else "Adapted as"
    return "Same series" if link.link_type == "same_franchise" else "Same creator"


def linked_works(conn: sqlite3.Connection, work_id: int) -> list[LinkedWork]:
    """Return works joined to this one by a **verified** link, strongest first.

    Unverified candidates from `match` are left out until `link_check` passes them.
    """
    rows = conn.execute(
        "SELECT from_work, to_work, link_type, source, confidence, verified FROM work_links "
        "WHERE verified = 1 AND (from_work = ? OR to_work = ?)",
        (work_id, work_id),
    ).fetchall()
    links = [
        Link(r["from_work"], r["to_work"], r["link_type"], r["source"], r["confidence"], True)
        for r in rows
    ]
    others = {(ln.to_work if ln.from_work == work_id else ln.from_work) for ln in links}
    works = works_by_ids(conn, list(others))
    order = {"adaptation_of": 0, "same_franchise": 1, "same_creator": 2}
    result = [
        LinkedWork(works[other], ln.link_type, _relation(ln, work_id), ln.confidence)
        for ln in links
        if (other := ln.to_work if ln.from_work == work_id else ln.from_work) in works
    ]
    return sorted(result, key=lambda lw: (order[lw.link_type], -lw.confidence, lw.work.id))


def similar_vibe(
    conn: sqlite3.Connection, work: Work, persona_id: int | None = None, k: int = 6
) -> list[SimilarWork]:
    """Return up to k works in the *other* medium that feel alike, excluding directly linked ones.

    Linked works have their own section, so results with a link score are dropped here. A work with
    no vibe vector (or no counterpart medium) has no vibe neighbours, so the list is empty.
    """
    target = _OTHER_MEDIUM.get(work.media_type)
    if target is None or not has_vector(conn, work.id):
        return []
    linked = {lw.work.id for lw in linked_works(conn, work.id)}
    recs = recommend.overlap(conn, work.id, target, persona_id, k=k + len(linked))
    recs = [r for r in recs if r.link == 0.0 and r.work_id not in linked][:k]
    works = works_by_ids(conn, [r.work_id for r in recs])
    return [SimilarWork(works[r.work_id], r) for r in recs if r.work_id in works]


def work_detail(
    conn: sqlite3.Connection, work_id: int, persona_id: int | None = None, k: int = 6
) -> WorkDetail | None:
    """Return the full Work page data, or None if the work doesn't exist."""
    work = get_work(conn, work_id)
    if work is None:
        return None
    return WorkDetail(
        work=work,
        card=get_card(conn, work_id),
        linked=linked_works(conn, work_id),
        similar=similar_vibe(conn, work, persona_id, k),
        has_vector=has_vector(conn, work_id),
    )
