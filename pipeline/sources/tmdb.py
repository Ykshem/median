"""TMDB (movies): search a title, fetch details, normalise to FetchedWork.

Auth is the v4 read token as a Bearer header only (never a query param), so it can't appear in a
URL, a log line or a cache file. Terms: non-commercial use, attribution on the Credits page.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from median.config import get_settings
from pipeline.sources import FetchedWork, SkipWork, clip_synopsis, normalize_title
from pipeline.sources.http import CachedClient

logger = logging.getLogger(__name__)

BASE_URL = "https://api.themoviedb.org"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
MAX_KEYWORDS = 12
MAX_DISCOVER_PAGES = 500  # TMDB caps list pagination at 500 pages (10,000 results)
DISCOVER_MIN_VOTES = 500  # skips obscure titles whose popularity is noise


def make_client(**kwargs: Any) -> CachedClient:
    """Return a TMDB client (stays well under ~40 requests per 10 s). Needs TMDB_API_READ_TOKEN."""
    token = get_settings().tmdb_api_read_token
    if not token:
        raise RuntimeError(
            "TMDB_API_READ_TOKEN is not set. Copy .env.example to .env and fill it in."
        )
    return CachedClient(
        "tmdb",
        BASE_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        min_interval=0.3,
        **kwargs,
    )


def _year(date: str | None) -> int | None:
    return int(date[:4]) if date and date[:4].isdigit() else None


def normalize_movie(raw: dict[str, Any]) -> FetchedWork:
    """Convert a `/movie/{id}?append_to_response=keywords,credits,external_ids` payload."""
    overview = (raw.get("overview") or "").strip()
    if not overview:
        raise SkipWork(f"TMDB {raw.get('id')} ({raw.get('title')}) has an empty overview")
    crew = raw.get("credits", {}).get("crew", [])
    keywords = raw.get("keywords", {}).get("keywords", [])
    poster = raw.get("poster_path")
    return FetchedWork(
        source="tmdb",
        external_id=str(raw["id"]),
        media_type="movie",
        title=raw["title"],
        year=_year(raw.get("release_date")),
        creators=[c["name"] for c in crew if c.get("job") == "Director"],
        synopsis=clip_synopsis(overview),
        subjects=[g["name"] for g in raw.get("genres", [])]
        + [k["name"] for k in keywords][:MAX_KEYWORDS],
        cover_url=f"{IMAGE_BASE}{poster}" if poster else None,
        popularity_raw=raw.get("popularity"),
    )


def get_movie(client: CachedClient, movie_id: int | str) -> FetchedWork:
    """Fetch details for one movie. Raises SkipWork if it has no overview."""
    raw = client.get_json(
        f"/3/movie/{movie_id}", {"append_to_response": "keywords,credits,external_ids"}
    )
    return normalize_movie(raw)


def find_movie(client: CachedClient, title: str, year: int) -> FetchedWork:
    """Return the movie matching `title` and `year` (exact match preferred).

    A non-exact pick is returned with `match_note` set so the resolution report can flag it.
    Raises SkipWork if the search finds nothing or the movie has no overview.
    """
    results = client.get_json("/3/search/movie", {"query": title, "year": year}).get("results", [])
    if not results:
        raise SkipWork(f"TMDB search found nothing for {title!r} ({year})")
    wanted = normalize_title(title)
    exact = [
        r
        for r in results
        if normalize_title(r.get("title", "")) == wanted and _year(r.get("release_date")) == year
    ]
    chosen = (exact or results)[0]
    work = get_movie(client, chosen["id"])
    if exact:
        return work
    note = (
        f"weak match for {title!r} ({year}): "
        f"picked {chosen.get('title')!r} ({chosen.get('release_date')})"
    )
    logger.warning(note)
    return dataclasses.replace(work, match_note=note)


def discover_ids(
    client: CachedClient,
    limit: int,
    *,
    min_votes: int = DISCOVER_MIN_VOTES,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[int]:
    """Return up to `limit` movie ids, most popular first (optionally within a release-year range).

    TMDB stops list pagination at 10,000 results, so bigger sets need `discover_sliced`.
    """
    ids: list[int] = []
    page = 1
    while len(ids) < limit and page <= MAX_DISCOVER_PAGES:
        params: dict[str, Any] = {
            "sort_by": "popularity.desc",
            "vote_count.gte": min_votes,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        }
        if start_year:
            params["primary_release_date.gte"] = f"{start_year}-01-01"
        if end_year:
            params["primary_release_date.lte"] = f"{end_year}-12-31"
        data = client.get_json("/3/discover/movie", params)
        ids.extend(r["id"] for r in data.get("results", []))
        if page >= data.get("total_pages", 1):
            break
        page += 1
    return ids[:limit]


def discover_sliced(
    client: CachedClient, limit: int, years: range, *, min_votes: int = DISCOVER_MIN_VOTES
) -> list[int]:
    """Return about `limit` popular movie ids, sliced per release year (for the full build)."""
    per_year = -(-limit // len(years))  # ceiling division
    seen: dict[int, None] = {}
    for year in years:
        for movie_id in discover_ids(
            client, per_year, min_votes=min_votes, start_year=year, end_year=year
        ):
            seen.setdefault(movie_id)
    return list(seen)[:limit]
