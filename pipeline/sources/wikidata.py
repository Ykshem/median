"""Wikidata SPARQL: which films are based on which books, and which share a series.

Properties (checked against the live service): P31 instance of, Q11424 film, P4947 TMDb movie ID,
P144 based on, P179 part of the series, P648 Open Library ID (we keep only works: ids ending "W").
Queries must finish within 60 s, so results are paged with LIMIT/OFFSET and the output is cached.
Coverage is partial (roughly a third of real adaptations), hence `match` and `link_check`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from median.config import get_settings
from pipeline.sources.http import CachedClient

logger = logging.getLogger(__name__)

BASE_URL = "https://query.wikidata.org"
PAGE_SIZE = 5000
MAX_PAGES = 40  # a safety stop: 200k rows

ADAPTATION_QUERY = """
SELECT ?tmdb ?ol WHERE {{
  ?film wdt:P31 wd:Q11424; wdt:P4947 ?tmdb; wdt:P144 ?book.
  ?book wdt:P648 ?ol.
  FILTER(STRENDS(?ol, "W"))
}} ORDER BY ?tmdb ?ol LIMIT {limit} OFFSET {offset}
"""

SERIES_QUERY = """
SELECT ?tmdb ?ol ?series WHERE {{
  ?film wdt:P31 wd:Q11424; wdt:P4947 ?tmdb; wdt:P179 ?series.
  ?book wdt:P179 ?series; wdt:P648 ?ol.
  FILTER(STRENDS(?ol, "W"))
}} ORDER BY ?tmdb ?ol ?series LIMIT {limit} OFFSET {offset}
"""


def make_client(**kwargs: Any) -> CachedClient:
    """Return a Wikidata client. Wikidata rejects generic user agents, so this names the app."""
    contact = get_settings().contact_email
    if not contact:
        raise RuntimeError("CONTACT_EMAIL is not set; Wikidata asks for it in the User-Agent.")
    return CachedClient(
        "wikidata",
        BASE_URL,
        headers={
            "User-Agent": f"Median/0.1 (class project; contact: {contact})",
            "Accept": "application/sparql-results+json",
        },
        min_interval=1.0,
        timeout=70.0,
        **kwargs,
    )


def _rows(client: CachedClient, template: str) -> Iterator[dict[str, Any]]:
    for page in range(MAX_PAGES):
        query = template.format(limit=PAGE_SIZE, offset=page * PAGE_SIZE)
        data = client.get_json("/sparql", {"query": query, "format": "json"})
        rows = data["results"]["bindings"]
        yield from rows
        if len(rows) < PAGE_SIZE:
            return
    logger.warning("Wikidata query hit the %d-page safety stop; results are truncated", MAX_PAGES)


def _value(row: dict[str, Any], name: str) -> str:
    return row[name]["value"]


def fetch_adaptation_pairs(client: CachedClient) -> list[tuple[int, str]]:
    """Return sorted unique (tmdb_movie_id, openlibrary_work_id): the film is based on the book."""
    pairs: set[tuple[int, str]] = set()
    for row in _rows(client, ADAPTATION_QUERY):
        tmdb = _value(row, "tmdb")
        if tmdb.isdigit():
            pairs.add((int(tmdb), _value(row, "ol")))
    logger.info("Wikidata: %d film/book adaptation pairs", len(pairs))
    return sorted(pairs)


def fetch_series_pairs(client: CachedClient) -> list[tuple[int, str, str]]:
    """Return sorted unique (tmdb_movie_id, openlibrary_work_id, series_id) for shared series."""
    triples: set[tuple[int, str, str]] = set()
    for row in _rows(client, SERIES_QUERY):
        tmdb = _value(row, "tmdb")
        if tmdb.isdigit():
            series = _value(row, "series").rsplit("/", 1)[-1]
            triples.add((int(tmdb), _value(row, "ol"), series))
    logger.info("Wikidata: %d film/book series pairs", len(triples))
    return sorted(triples)
