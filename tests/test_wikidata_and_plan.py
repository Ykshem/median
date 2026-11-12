from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from median import config
from pipeline.sample_plan import SamplePlan, SampleSpec, bare_id, select_sample
from pipeline.sources import tmdb, wikidata


def _bindings(*rows: dict[str, str]) -> dict:
    return {"results": {"bindings": [{k: {"value": v} for k, v in row.items()} for row in rows]}}


def _wikidata_client(tmp_path: Path, pages: list[dict], seen: list[str] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        if seen is not None:
            seen.append(query)
        return httpx.Response(200, json=pages[offset // wikidata.PAGE_SIZE])

    return wikidata.make_client(
        transport=httpx.MockTransport(handler), cache_dir=tmp_path, sleep=lambda s: None
    )


# Wikidata ----------------------------------------------------------------------------------------


def test_adaptation_pairs_are_parsed_deduplicated_and_sorted(tmp_path: Path) -> None:
    page = _bindings(
        {"tmdb": "694", "ol": "OL81633W"},
        {"tmdb": "12", "ol": "OL5W"},
        {"tmdb": "694", "ol": "OL81633W"},  # a repeated row
        {"tmdb": "not-a-number", "ol": "OL9W"},  # a malformed id
    )
    with _wikidata_client(tmp_path, [page]) as client:
        pairs = wikidata.fetch_adaptation_pairs(client)
    assert pairs == [(12, "OL5W"), (694, "OL81633W")]


def test_series_pairs_carry_the_series_id(tmp_path: Path) -> None:
    page = _bindings({"tmdb": "1", "ol": "OL2W", "series": "http://www.wikidata.org/entity/Q42"})
    with _wikidata_client(tmp_path, [page]) as client:
        assert wikidata.fetch_series_pairs(client) == [(1, "OL2W", "Q42")]


def test_results_are_paged_until_a_short_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wikidata, "PAGE_SIZE", 2)
    pages = [
        _bindings({"tmdb": "1", "ol": "OL1W"}, {"tmdb": "2", "ol": "OL2W"}),
        _bindings({"tmdb": "3", "ol": "OL3W"}),
    ]
    seen: list[str] = []
    with _wikidata_client(tmp_path, pages, seen) as client:
        assert len(wikidata.fetch_adaptation_pairs(client)) == 3
    assert len(seen) == 2 and "OFFSET 0" in seen[0] and "OFFSET 2" in seen[1]
    assert all("LIMIT 2" in q for q in seen)


def test_queries_filter_to_works_and_films() -> None:
    for template in (wikidata.ADAPTATION_QUERY, wikidata.SERIES_QUERY):
        assert "wd:Q11424" in template and "P4947" in template and "P648" in template
        assert 'STRENDS(?ol, "W")' in template  # works only, not editions or authors


def test_wikidata_client_names_the_app(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_bindings())

    client = wikidata.make_client(
        transport=httpx.MockTransport(handler), cache_dir=tmp_path, sleep=lambda s: None
    )
    with client:
        wikidata.fetch_adaptation_pairs(client)
    assert seen[0].headers["User-Agent"] == "Median/0.1 (class project; contact: test@example.com)"


def test_wikidata_requires_a_contact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTACT_EMAIL", "")
    config.get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="CONTACT_EMAIL"):
        wikidata.make_client()


# TMDB discover -----------------------------------------------------------------------------------


def _discover_client(tmp_path: Path, total_pages: int, per_page: int = 3, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        if seen is not None:
            seen.append(dict(request.url.params))
        ids = [page * 100 + i for i in range(per_page)]
        return httpx.Response(
            200, json={"results": [{"id": i} for i in ids], "total_pages": total_pages}
        )

    return tmdb.make_client(
        transport=httpx.MockTransport(handler), cache_dir=tmp_path, sleep=lambda s: None
    )


def test_discover_collects_pages_in_order_and_trims_to_the_limit(tmp_path: Path) -> None:
    seen: list = []
    with _discover_client(tmp_path, total_pages=10, seen=seen) as client:
        ids = tmdb.discover_ids(client, 7)
    assert ids == [100, 101, 102, 200, 201, 202, 300]
    assert len(seen) == 3
    assert seen[0]["sort_by"] == "popularity.desc" and seen[0]["vote_count.gte"] == "500"


def test_discover_stops_at_the_last_page(tmp_path: Path) -> None:
    with _discover_client(tmp_path, total_pages=2) as client:
        assert len(tmdb.discover_ids(client, 100)) == 6


def test_discover_year_range_sets_release_date_bounds(tmp_path: Path) -> None:
    seen: list = []
    with _discover_client(tmp_path, total_pages=1, seen=seen) as client:
        tmdb.discover_ids(client, 2, start_year=1999, end_year=2001)
    assert seen[0]["primary_release_date.gte"] == "1999-01-01"
    assert seen[0]["primary_release_date.lte"] == "2001-12-31"


def test_discover_sliced_covers_every_year_without_duplicates(tmp_path: Path) -> None:
    seen: list = []
    with _discover_client(tmp_path, total_pages=1, per_page=2, seen=seen) as client:
        ids = tmdb.discover_sliced(client, 6, range(2000, 2003))
    years = [p["primary_release_date.gte"][:4] for p in seen]
    assert years == ["2000", "2001", "2002"]
    assert len(ids) == len(set(ids)) <= 6


# Sample selection --------------------------------------------------------------------------------


def _select(spec: SampleSpec, **overrides):
    wikidata_pairs = [(1000 + i, f"OL{i}W") for i in range(1, 31)]  # 30 books, 1 film each
    base = dict(
        forced_movies=[],
        forced_books=[],
        forced_pairs=[],
        wikidata_pairs=wikidata_pairs,
        reading_counts={f"/works/OL{i}W": 10_000 - i for i in range(1, 400)},
        has_description=lambda ids: {i: True for i in ids},
        is_fiction=lambda ids: {i: True for i in ids},
        discover_movies=lambda n: list(range(5000, 5000 + n)),
    )
    base.update(overrides)
    return select_sample(spec, **base)


SMALL = SampleSpec(movies=60, books=50, min_pairs=10, pair_buffer=1.0, book_candidates=200)


def test_plan_hits_the_requested_sizes_and_pair_count() -> None:
    plan = _select(SMALL)
    assert len(plan.movies) == 60 and len(plan.books) == 50
    assert len(plan.pairs) == 10
    assert len(set(plan.movies)) == 60 and len(set(plan.books)) == 50
    # every paired film and book is in the plan
    assert all(m in plan.movies and b in plan.books for m, b in plan.pairs)


def test_pairs_are_taken_from_the_most_read_books_first() -> None:
    plan = _select(SMALL)
    assert {b for _, b in plan.pairs} == {f"OL{i}W" for i in range(1, 11)}


def test_forced_works_and_pairs_are_always_included() -> None:
    plan = _select(
        SMALL,
        forced_movies=[777],
        forced_books=["OL399W"],
        forced_pairs=[(888, "OL398W")],
    )
    assert 777 in plan.movies and "OL399W" in plan.books
    assert (888, "OL398W") in plan.pairs and 888 in plan.movies and "OL398W" in plan.books
    assert len(plan.pairs) == 10  # forced pairs count towards the target


def test_books_without_a_description_are_skipped() -> None:
    no_desc = {f"OL{i}W" for i in range(1, 6)}
    plan = _select(SMALL, has_description=lambda ids: {i: i not in no_desc for i in ids})
    assert not no_desc & set(plan.books)
    assert not no_desc & {b for _, b in plan.pairs}
    assert len(plan.pairs) == 10  # replaced by the next most popular pair books


def test_one_book_cannot_flood_the_pairs() -> None:
    pairs = [(2000 + i, "OL1W") for i in range(20)] + [(3000 + i, f"OL{i}W") for i in range(2, 30)]
    plan = _select(SMALL, wikidata_pairs=pairs)
    assert sum(1 for _, b in plan.pairs if b == "OL1W") == 2  # max_films_per_book


def test_plan_is_deterministic_and_round_trips_through_json(tmp_path: Path) -> None:
    first, second = _select(SMALL), _select(SMALL)
    assert first == second
    path = tmp_path / "plan.json"
    first.save(path)
    loaded = SamplePlan.load(path)
    assert loaded.movies == first.movies and loaded.books == first.books
    assert loaded.pairs == first.pairs


def test_reading_counts_accept_full_keys_or_bare_ids() -> None:
    assert bare_id("/works/OL5W") == bare_id("OL5W") == "OL5W"


def test_movie_fill_does_not_duplicate_pair_films() -> None:
    # discover returns the paired films too; they must not be counted twice
    plan = _select(SMALL, discover_movies=lambda n: [1001, 1002, *range(7000, 7000 + n)])
    assert len(plan.movies) == len(set(plan.movies)) == 60


# Fiction-only books (ADR 0002) ------------------------------------------------------------------


def _fiction_except(*banned: str):
    return lambda ids: {i: i not in banned for i in ids}


def test_nonfiction_books_never_enter_the_plan() -> None:
    banned = {f"OL{i}W" for i in range(1, 40, 2)}  # every other book, pair partners included
    plan = _select(SMALL, is_fiction=_fiction_except(*banned))
    assert not set(plan.books) & banned
    assert not {b for _, b in plan.pairs} & banned
    assert len(plan.books) == SMALL.books  # the fill goes further down the list instead


def test_a_nonfiction_pair_partner_is_replaced_by_the_next_most_read() -> None:
    plan = _select(SMALL, is_fiction=_fiction_except("OL1W"))
    assert "OL1W" not in plan.books and (1001, "OL1W") not in plan.pairs
    assert "OL11W" in plan.books  # pairs run through books by reading count, so OL11W is still in


def test_a_forced_nonfiction_book_and_its_pairs_are_dropped(caplog) -> None:
    plan = _select(
        SMALL,
        forced_books=["OL900W", "OL901W"],
        forced_pairs=[(9001, "OL900W")],
        is_fiction=_fiction_except("OL900W"),
    )
    assert "OL900W" not in plan.books and (9001, "OL900W") not in plan.pairs
    assert "OL901W" in plan.books  # a forced fiction book still goes in
    assert 9001 not in plan.movies
    assert "OL900W" in caplog.text


def test_the_fiction_check_is_only_asked_about_books_that_could_be_selected() -> None:
    asked: list[str] = []

    def is_fiction(ids):
        asked.extend(ids)
        return {i: True for i in ids}

    no_desc = {"OL2W", "OL3W"}
    _select(
        SMALL, is_fiction=is_fiction, has_description=lambda ids: {i: i not in no_desc for i in ids}
    )
    assert not set(asked) & no_desc  # the model is never paid to judge an unusable book
    assert len(asked) == len(set(asked))
