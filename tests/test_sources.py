from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from median import config
from pipeline.sources import MAX_SYNOPSIS_CHARS, SkipWork, clip_synopsis, normalize_title, tmdb
from pipeline.sources import openlibrary as ol
from tests.fixtures.source_payloads import (
    ol_doc,
    ol_search,
    ol_work,
    tmdb_details,
    tmdb_search,
)

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """An httpx MockTransport handler that serves canned JSON by path and records requests."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        payload = self.routes[request.url.path]
        if isinstance(payload, httpx.Response):
            return payload
        return httpx.Response(200, json=payload)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def _tmdb_client(rec: Recorder, tmp_path: Path, **kw: object):
    return tmdb.make_client(
        transport=rec.transport, cache_dir=tmp_path / "cache", sleep=lambda s: None, **kw
    )


def _ol_client(rec: Recorder, tmp_path: Path, **kw: object):
    return ol.make_client(
        transport=rec.transport, cache_dir=tmp_path / "cache", sleep=lambda s: None, **kw
    )


# Helpers -----------------------------------------------------------------------------------------


def test_normalize_title_ignores_case_punctuation_and_ampersand() -> None:
    assert normalize_title("Pride & Prejudice") == normalize_title("Pride and Prejudice")
    assert normalize_title("Charlotte's Web") == normalize_title("Charlotte’s Web")
    assert normalize_title("Mad Max: Fury Road") == "mad max fury road"


def test_clip_synopsis_bounds_length() -> None:
    assert clip_synopsis("a  b\n c") == "a b c"
    clipped = clip_synopsis("x" * (MAX_SYNOPSIS_CHARS + 50))
    assert len(clipped) == MAX_SYNOPSIS_CHARS + 1 and clipped.endswith("…")


# TMDB --------------------------------------------------------------------------------------------


def test_tmdb_normalizes_details(tmp_path: Path) -> None:
    rec = Recorder({"/3/movie/694": tmdb_details()})
    with _tmdb_client(rec, tmp_path) as client:
        work = tmdb.get_movie(client, 694)
    assert (work.source, work.external_id, work.media_type) == ("tmdb", "694", "movie")
    assert (work.title, work.year, work.creators) == ("The Shining", 1980, ["Stanley Kubrick"])
    assert work.subjects == ["Horror", "Thriller", "hotel", "isolation"]
    assert work.cover_url == "https://image.tmdb.org/t/p/w342/poster.jpg"
    assert rec.requests[0].url.params["append_to_response"] == "keywords,credits,external_ids"


def test_tmdb_prefers_exact_title_and_year_over_a_more_popular_result(tmp_path: Path) -> None:
    rec = Recorder(
        {
            "/3/search/movie": tmdb_search(
                (2, "The Shining", "1997-04-27"), (694, "The Shining", "1980-05-23")
            ),
            "/3/movie/694": tmdb_details(),
        }
    )
    with _tmdb_client(rec, tmp_path) as client:
        work = tmdb.find_movie(client, "The Shining", 1980)
    assert work.external_id == "694"
    assert work.match_note is None


def test_tmdb_weak_match_is_flagged(tmp_path: Path) -> None:
    rec = Recorder(
        {
            "/3/search/movie": tmdb_search((7, "Shining Through", "1992-01-31")),
            "/3/movie/7": tmdb_details(7, "Shining Through", "1992-01-31"),
        }
    )
    with _tmdb_client(rec, tmp_path) as client:
        work = tmdb.find_movie(client, "The Shining", 1980)
    assert work.match_note and "weak match" in work.match_note


def test_tmdb_skips_empty_overview_and_empty_search(tmp_path: Path) -> None:
    rec = Recorder(
        {
            "/3/search/movie": tmdb_search((5, "Silent", "2000-01-01")),
            "/3/movie/5": tmdb_details(5, "Silent", "2000-01-01", overview="  "),
        }
    )
    with _tmdb_client(rec, tmp_path) as client, pytest.raises(SkipWork, match="empty overview"):
        tmdb.find_movie(client, "Silent", 2000)
    empty = Recorder({"/3/search/movie": {"results": []}})
    with _tmdb_client(empty, tmp_path / "b") as client, pytest.raises(SkipWork, match="nothing"):
        tmdb.find_movie(client, "Nope", 1999)


def test_tmdb_uses_bearer_header_and_never_leaks_the_token(tmp_path: Path) -> None:
    rec = Recorder({"/3/movie/694": tmdb_details()})
    with _tmdb_client(rec, tmp_path) as client:
        tmdb.get_movie(client, 694)
    request = rec.requests[0]
    assert request.headers["Authorization"] == "Bearer test-tmdb-token"
    assert "test-tmdb-token" not in str(request.url)
    for cache_file in (tmp_path / "cache").rglob("*"):
        assert "test-tmdb-token" not in cache_file.name
        if cache_file.is_file():
            assert "test-tmdb-token" not in cache_file.read_text(encoding="utf-8")


def test_tmdb_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_API_READ_TOKEN", "")
    config.get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="TMDB_API_READ_TOKEN"):
        tmdb.make_client()


def test_cache_hit_makes_no_network_call(tmp_path: Path) -> None:
    rec = Recorder({"/3/movie/694": tmdb_details()})
    with _tmdb_client(rec, tmp_path) as client:
        first = tmdb.get_movie(client, 694)
        second = tmdb.get_movie(client, 694)
    assert first == second
    assert len(rec.requests) == 1
    # A fresh client on the same cache directory also needs no request.
    with _tmdb_client(Recorder({}), tmp_path) as client:
        assert tmdb.get_movie(client, 694) == first
    cached = next((tmp_path / "cache" / "tmdb").glob("*.json"))
    assert json.loads(cached.read_text(encoding="utf-8"))["path"] == "/3/movie/694"


def test_requests_are_throttled(tmp_path: Path) -> None:
    sleeps: list[float] = []
    rec = Recorder({"/3/movie/1": tmdb_details(1), "/3/movie/2": tmdb_details(2)})
    client = tmdb.make_client(
        transport=rec.transport, cache_dir=tmp_path, sleep=sleeps.append, clock=lambda: 100.0
    )
    with client:
        tmdb.get_movie(client, 1)
        assert sleeps == []  # first request goes straight out
        tmdb.get_movie(client, 2)
    assert sleeps == [pytest.approx(0.3)]


def test_rate_limit_429_waits_then_retries(tmp_path: Path) -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json=tmdb_details())

    client = tmdb.make_client(
        transport=httpx.MockTransport(handler), cache_dir=tmp_path, sleep=sleeps.append
    )
    with client:
        assert tmdb.get_movie(client, 694).title == "The Shining"
    assert calls["n"] == 2 and 7.0 in sleeps


def test_persistent_429_gives_up(tmp_path: Path) -> None:
    client = tmdb.make_client(
        transport=httpx.MockTransport(lambda r: httpx.Response(429)),
        cache_dir=tmp_path,
        sleep=lambda s: None,
    )
    with client, pytest.raises(RuntimeError, match="rate-limited"):
        tmdb.get_movie(client, 1)


# Open Library ------------------------------------------------------------------------------------


def test_ol_normalizes_search_doc_and_work(tmp_path: Path) -> None:
    rec = Recorder({"/search.json": ol_search(ol_doc()), "/works/OL81634W.json": ol_work()})
    with _ol_client(rec, tmp_path) as client:
        work = ol.find_book(client, "The Shining", "Stephen King")
    assert (work.source, work.external_id, work.media_type) == ("openlibrary", "OL81634W", "book")
    assert (work.title, work.year, work.creators) == ("The Shining", 1977, ["Stephen King"])
    assert work.subjects == ["Horror", "Hotels", "Fiction"]
    assert work.cover_url == "https://covers.openlibrary.org/b/id/12345-M.jpg"
    assert work.match_note is None


def test_ol_picks_the_matching_author_not_the_first_doc(tmp_path: Path) -> None:
    rec = Recorder(
        {
            "/search.json": ol_search(
                ol_doc("/works/OL1W", "The Shining", ("Someone Else",), 2001),
                ol_doc("/works/OL81634W"),
            ),
            "/works/OL81634W.json": ol_work(),
        }
    )
    with _ol_client(rec, tmp_path) as client:
        assert ol.find_book(client, "The Shining", "Stephen King").external_id == "OL81634W"


def test_ol_drops_edition_noise_from_the_author_list(tmp_path: Path) -> None:
    noisy = ol_doc(
        "/works/OL35616W", "Kindred", ("Octavia E. Butler", "SparkNotes", "Tomi Adeyemi")
    )
    rec = Recorder({"/search.json": ol_search(noisy), "/works/OL35616W.json": ol_work()})
    with _ol_client(rec, tmp_path) as client:
        work = ol.find_book(client, "Kindred", "Octavia E. Butler")
    assert work.creators == ["Octavia E. Butler"]


def test_ol_weak_match_is_flagged(tmp_path: Path) -> None:
    rec = Recorder(
        {
            "/search.json": ol_search(ol_doc("/works/OL9W", "Shining Girls", ("Lauren Beukes",))),
            "/works/OL9W.json": ol_work(),
        }
    )
    with _ol_client(rec, tmp_path) as client:
        work = ol.find_book(client, "The Shining", "Stephen King")
    assert work.match_note and "weak match" in work.match_note


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Plain text.", "Plain text."),
        ({"type": "/type/text", "value": "From a dict."}, "From a dict."),
        ("Real blurb.\r\n----------\nSource: somewhere", "Real blurb."),
        ("See [the author](https://example.org/a) for more.", "See the author for more."),
        (None, ""),
        (123, ""),
    ],
)
def test_clean_description(raw: object, expected: str) -> None:
    assert ol.clean_description(raw) == expected


def test_ol_skips_a_work_with_no_description(tmp_path: Path) -> None:
    rec = Recorder({"/search.json": ol_search(ol_doc()), "/works/OL81634W.json": ol_work(None)})
    with _ol_client(rec, tmp_path) as client, pytest.raises(SkipWork, match="no description"):
        ol.find_book(client, "The Shining", "Stephen King")


def test_ol_user_agent_names_the_app_and_the_configured_contact(tmp_path: Path) -> None:
    rec = Recorder({"/search.json": ol_search(ol_doc()), "/works/OL81634W.json": ol_work()})
    with _ol_client(rec, tmp_path) as client:
        ol.find_book(client, "The Shining", "Stephen King")
    agent = rec.requests[0].headers["User-Agent"]
    assert agent == "Median/0.1 (class project; contact: test@example.com)"


def test_ol_refuses_to_run_without_a_contact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTACT_EMAIL", "")
    config.get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="CONTACT_EMAIL"):
        ol.make_client()
