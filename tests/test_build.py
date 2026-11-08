"""The whole build, offline: synthetic dumps, fake TMDB/Wikidata/Open Library, fake LLM clients."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from median import llm
from median.db import connect, get_meta
from pipeline import build
from pipeline.sample_plan import SamplePlan, SampleSpec
from pipeline.sources import openlibrary, tmdb, wikidata
from tests.fakes import FakeBatchClient, FakeOpenAI, embedding_response
from tests.fixtures.source_payloads import tmdb_details
from tests.test_catalog import _line, _write_dump
from tests.test_vibe_cards import GOOD_CARD

SPEC = SampleSpec(
    movies=8, books=8, min_pairs=2, pair_buffer=1.0, book_candidates=20, min_description_chars=20
)
LONG = "A story long enough to count as a real description. "


def _tmdb_payload(movie_id: int) -> dict:
    payload = tmdb_details(
        movie_id,
        f"Movie {movie_id}",
        f"{2000 + movie_id}-01-01",
        overview=f"Overview of movie {movie_id}.",
    )
    payload["popularity"] = movie_id * 10.0
    crew = [{"name": "A Director", "job": "Director"}]
    if movie_id == 5:  # credited to the novel's author, but NOT in Wikidata: `match` must find it
        crew.append({"name": "Author Five", "job": "Novel"})
    payload["credits"] = {"crew": crew}
    return payload


def _client_factory(make, handler, tmp_path: Path, name: str):
    return lambda: make(
        transport=httpx.MockTransport(handler), cache_dir=tmp_path / name, sleep=lambda s: None
    )


def _tmdb_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/3/discover/movie":
        return httpx.Response(
            200, json={"results": [{"id": i} for i in range(1, 11)], "total_pages": 1}
        )
    if path == "/3/search/movie":
        return httpx.Response(200, json={"results": []})  # the real review set isn't in this world
    return httpx.Response(200, json=_tmdb_payload(int(path.rsplit("/", 1)[-1])))


def _wikidata_handler(request: httpx.Request) -> httpx.Response:
    query = request.url.params["query"]
    if "P144" in query:  # adaptations: three films, three books
        rows = [{"tmdb": {"value": str(i)}, "ol": {"value": f"OL{i}W"}} for i in (1, 2, 3)]
    else:  # series: movie 4 shares a series with book 4
        rows = [
            {
                "tmdb": {"value": "4"},
                "ol": {"value": "OL4W"},
                "series": {"value": "http://www.wikidata.org/entity/Q7"},
            }
        ]
    return httpx.Response(200, json={"results": {"bindings": rows}})


def _write_world(tmp_path: Path) -> Path:
    """Twelve books (OL12W has no description), reading counts favouring low numbers."""
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    works = []
    for i in range(1, 13):
        record = {
            "title": f"Book {i}",
            "authors": [{"author": {"key": f"/authors/OL{i}A"}}],
            "first_publish_date": str(1990 + i % 5),
            "subjects": ["Fiction"],
        }
        if i != 12:
            record["description"] = LONG + f"Book {i}."
        works.append(_line("work", f"/works/OL{i}W", record))
    _write_dump(dumps / "works.gz", works)
    names = {i: f"Author {i}" for i in range(1, 13)} | {5: "Author Five"}
    _write_dump(
        dumps / "authors.gz",
        [_line("author", f"/authors/OL{i}A", {"name": names[i]}) for i in range(1, 13)],
    )
    log = [
        f"/works/OL{i}W\t\\N\tAlready Read\t2026-01-01" for i in range(1, 13) for _ in range(13 - i)
    ]
    _write_dump(dumps / "log.gz", log)
    gold = tmp_path / "gold.csv"
    gold.write_text("book_ol_id,movie_tmdb_id\nOL9W,9\n", encoding="utf-8")
    return dumps


def _card_or_verdict(request: dict) -> str:
    prompt = request["params"]["messages"][0]["content"]
    if "Decide whether the film is an adaptation" in prompt:
        return json.dumps({"is_adaptation": True, "confidence": 0.9, "reason": "credited"})
    return json.dumps(GOOD_CARD)


@pytest.fixture
def world(db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dumps = _write_world(tmp_path)
    paths = {
        "reading-log": dumps / "log.gz",
        "authors": dumps / "authors.gz",
        "works": dumps / "works.gz",
    }
    monkeypatch.setattr(build, "_dumps", lambda ctx, *names: {n: paths[n] for n in names})
    monkeypatch.setattr(
        build, "_tmdb_client", _client_factory(tmdb.make_client, _tmdb_handler, tmp_path, "t")
    )
    monkeypatch.setattr(
        build,
        "_wikidata_client",
        _client_factory(wikidata.make_client, _wikidata_handler, tmp_path, "w"),
    )
    monkeypatch.setattr(
        build,
        "_openlibrary_client",
        _client_factory(
            openlibrary.make_client, lambda r: httpx.Response(200, json={"docs": []}), tmp_path, "o"
        ),
    )
    anthropic = FakeBatchClient(_card_or_verdict)
    openai = FakeOpenAI([embedding_response(16), embedding_response(16)])
    monkeypatch.setattr(llm, "_anthropic_client", lambda: anthropic)
    monkeypatch.setattr(llm, "_openai_client", lambda: openai)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    # keep the verdict cache out of the real data/ directory
    monkeypatch.setattr(build.link_step, "CACHE_PATH", tmp_path / "verdicts.json")
    ctx = build.BuildContext(
        conn=db,
        sample=True,
        spec=SPEC,
        dumps_dir=dumps,
        cache_dir=tmp_path / "cache",
        plan_path=tmp_path / "plan.json",
        gold_path=tmp_path / "gold.csv",
        vibes_path=tmp_path / "vibes.csv",  # absent, so the real vibe file never leaks in
    )
    return ctx, anthropic, openai


def _count(db: sqlite3.Connection, sql: str) -> int:
    return db.execute(sql).fetchone()[0]


def test_select_sample_builds_the_expected_plan(world) -> None:
    ctx, _, _ = world
    build.run_steps(ctx, ["select_sample"])
    plan = SamplePlan.load(ctx.plan_path)
    assert plan.pairs == [
        (9, "OL9W"),
        (1, "OL1W"),
    ]  # the gold pair, then the most-read Wikidata pair
    assert plan.books == ["OL9W", "OL1W", "OL2W", "OL3W", "OL4W", "OL5W", "OL6W", "OL7W"]
    assert plan.movies == [9, 1, 2, 3, 4, 5, 6, 7]
    assert "OL12W" not in plan.books  # no description


def test_the_non_llm_steps_load_link_and_match(world) -> None:
    ctx, anthropic, openai = world
    build.run_steps(ctx, build.select_steps(None, "match"))
    db = ctx.conn
    assert _count(db, "SELECT COUNT(*) FROM works WHERE media_type = 'movie'") == 8
    assert _count(db, "SELECT COUNT(*) FROM works WHERE media_type = 'book'") == 8
    book = db.execute("SELECT * FROM works WHERE title = 'Book 5'").fetchone()
    assert json.loads(book["creators"]) == ["Author Five"] and book["popularity_raw"] == 8.0
    links = {
        (r["link_type"], r["source"], r["verified"]): r["n"]
        for r in db.execute(
            "SELECT link_type, source, verified, COUNT(*) n FROM work_links GROUP BY 1, 2, 3"
        )
    }
    assert links == {
        ("adaptation_of", "wikidata", 1): 3,
        ("same_franchise", "wikidata", 1): 1,
        ("adaptation_of", "tmdb_credits", 0): 1,  # movie 5 <- Author Five's only book
    }
    assert anthropic.created == [] and _count(db, "SELECT COUNT(*) FROM llm_usage") == 0


def test_the_full_build_ends_with_verified_links_cards_vectors_and_a_clean_report(world) -> None:
    ctx, anthropic, openai = world
    build.run_steps(ctx, list(build.STEPS))
    db = ctx.conn
    assert ctx.problems == []
    assert _count(db, "SELECT COUNT(*) FROM work_links WHERE verified = 0") == 0
    assert _count(db, "SELECT COUNT(*) FROM work_links WHERE link_type = 'adaptation_of'") == 4
    assert _count(db, "SELECT COUNT(*) FROM vibe_cards") == 16
    assert _count(db, "SELECT COUNT(*) FROM vec_vibes") == 16
    assert _count(db, "SELECT COUNT(*) FROM works WHERE popularity IS NOT NULL") == 16
    assert get_meta(db, "embedding_model") and get_meta(db, "catalog_built_at")
    purposes = {r["purpose"] for r in db.execute("SELECT DISTINCT purpose FROM llm_usage")}
    assert purposes == {"link_check", "vibe_card", "embed"}
    assert len(anthropic.created) == 2  # one link_check batch, one vibe-card batch


def test_a_second_run_does_no_new_work_and_costs_nothing(world) -> None:
    ctx, anthropic, openai = world
    build.run_steps(ctx, list(build.STEPS))
    spent = _count(ctx.conn, "SELECT COUNT(*) FROM llm_usage")
    batches, embeds = len(anthropic.created), len(openai.calls)
    build.run_steps(ctx, list(build.STEPS))
    assert (len(anthropic.created), len(openai.calls)) == (batches, embeds)
    assert _count(ctx.conn, "SELECT COUNT(*) FROM llm_usage") == spent
    assert _count(ctx.conn, "SELECT COUNT(*) FROM works") == 16


def test_an_llm_step_that_would_cost_more_than_max_usd_is_refused(world) -> None:
    ctx, anthropic, _ = world
    build.run_steps(ctx, build.select_steps(None, "match"))
    ctx.max_usd = 0.0000001
    with pytest.raises(build.BuildError, match="max-usd"):
        build.run_steps(ctx, ["link_check"])
    with pytest.raises(build.BuildError, match="max-usd"):
        build.run_steps(ctx, ["vibe_cards"])
    assert anthropic.created == []


def test_targets_not_met_are_reported_and_make_the_build_fail(world) -> None:
    ctx, _, _ = world
    build.run_steps(ctx, build.select_steps(None, "match"))
    ctx.spec = SampleSpec(movies=500, books=500, min_pairs=50)
    build.run_steps(ctx, ["index", "report"])
    assert any("only 8 movies" in p for p in ctx.problems)


# CLI ---------------------------------------------------------------------------------------------


def test_step_selection() -> None:
    assert build.select_steps(None, None) == list(build.STEPS)
    assert build.select_steps("embed", None) == ["embed"]
    assert build.select_steps(None, "match") == [
        "select_sample",
        "fetch_tmdb",
        "load_openlibrary",
        "wikidata_links",
        "match",
    ]
    assert set(build.LLM_STEPS) <= set(build.STEPS)
    assert (
        build.STEPS.index("match")
        < build.STEPS.index("link_check")
        < build.STEPS.index("vibe_cards")
    )
    assert build.STEPS.index("vibe_cards") < build.STEPS.index("embed") < build.STEPS.index("index")


def test_a_full_build_is_refused_without_yes(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(build, "init_db", lambda: pytest.fail("must not touch the database"))
    with caplog.at_level("ERROR"):
        assert build.main(["--until", "match"]) == 2
    assert "costs about $5" in caplog.text


def test_the_sample_spec_matches_the_documented_targets() -> None:
    assert (build.SAMPLE_SPEC.movies, build.SAMPLE_SPEC.books, build.SAMPLE_SPEC.min_pairs) == (
        1000,
        1000,
        100,
    )
    assert build.FULL_SPEC.movies == 20_000 and build.FULL_SPEC.min_description_chars == 0


def test_main_runs_the_chosen_steps_and_reports_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    monkeypatch.setitem(build.STEP_FUNCS, "index", lambda ctx: seen.append("index"))

    def failing_report(ctx: build.BuildContext) -> None:
        seen.append("report")
        ctx.problems = ["something is short"]

    monkeypatch.setitem(build.STEP_FUNCS, "report", failing_report)
    assert build.main(["--sample", "--step", "index"]) == 0
    assert build.main(["--sample", "--step", "report"]) == 1
    assert seen == ["index", "report"]
    conn = connect()  # main() created the DB at the isolated test path
    conn.close()


def test_vibe_gold_works_are_forced_into_the_sample(world) -> None:
    ctx, _, _ = world
    ctx.vibes_path.write_text(
        "seed_ref,target_ref,seed_title,target_title,why,status\n"
        "tmdb:10,ol:OL12W,a,b,because,proposed\n",
        encoding="utf-8",
    )
    build.run_steps(ctx, ["select_sample"])
    plan = SamplePlan.load(ctx.plan_path)
    assert 10 in plan.movies and "OL12W" in plan.books
    assert len(plan.movies) == 8 and len(plan.books) == 8  # still the requested size


# regular (non-batch) API mode --------------------------------------------------------------------


def test_the_api_flag_selects_how_bulk_requests_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, object]] = []

    def capture(ctx: build.BuildContext) -> None:
        seen.append((ctx.api, ctx.dispatch))

    monkeypatch.setitem(build.STEP_FUNCS, "index", capture)
    assert build.main(["--sample", "--step", "index"]) == 0
    assert build.main(["--sample", "--step", "index", "--api", "regular"]) == 0
    assert seen == [("batch", llm.run_batch), ("regular", llm.run_parallel)]


def test_the_default_stays_batch_because_it_is_half_price() -> None:
    assert build.BuildContext(conn=None, sample=True, spec=SPEC).use_batch is True


def test_a_full_offline_build_through_the_regular_api_makes_no_batches(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fakes import FakeConcurrentClient

    ctx, batch_client, openai = world

    def answer(kw) -> str:
        return _card_or_verdict({"params": {"messages": kw["messages"]}})

    concurrent = FakeConcurrentClient(answer)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: concurrent)
    ctx.api = "regular"
    build.run_steps(ctx, list(build.STEPS))
    assert ctx.problems == []
    assert batch_client.created == []  # the Batches API was never touched
    assert len(concurrent.calls) == 16 + 1  # one card per work, and the one link check
    assert _count(ctx.conn, "SELECT COUNT(*) FROM vibe_cards") == 16
    assert _count(ctx.conn, "SELECT COUNT(*) FROM work_links WHERE verified = 0") == 0
    assert _count(ctx.conn, "SELECT COUNT(*) FROM meta WHERE key LIKE 'batch:%'") == 0


def test_the_regular_mode_guard_uses_the_full_price(world, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _, _ = world
    build.run_steps(ctx, build.select_steps(None, "match"))
    est_batch = vibe_cards_estimate(ctx, batch=True)
    est_regular = vibe_cards_estimate(ctx, batch=False)
    assert est_regular == pytest.approx(est_batch * 2)


def vibe_cards_estimate(ctx: build.BuildContext, *, batch: bool) -> float:
    template = build.vibe_cards.load_template()
    ids = [r["id"] for r in ctx.conn.execute("SELECT id FROM works")]
    prompts = [
        build.vibe_cards.render_prompt(template, build.vibe_cards.load_work(ctx.conn, i))
        for i in ids
    ]
    return build.vibe_cards.estimate(prompts, batch=batch).usd


def test_an_out_of_credit_api_error_exits_cleanly_with_code_3(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    import openai

    def no_credit(ctx: build.BuildContext) -> None:
        raise openai.RateLimitError(
            "You have no credits remaining",
            response=httpx.Response(429, request=httpx.Request("POST", "https://x.invalid")),
            body=None,
        )

    monkeypatch.setitem(build.STEP_FUNCS, "embed", no_credit)
    with caplog.at_level("ERROR"):
        assert build.main(["--sample", "--step", "embed"]) == 3
    assert "openai API error 429" in caplog.text
