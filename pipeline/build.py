"""Build the catalog database, one idempotent and resumable step at a time.

    uv run python -m pipeline.build --sample                  # the ~2,000-work dev catalog (~$0.20)
    uv run python -m pipeline.build --sample --until match    # everything that needs no LLM
    uv run python -m pipeline.build --sample --step report    # one step
    uv run python -m pipeline.build --yes                     # FULL catalog (~$5): ask first

Steps run in this order (ARCHITECTURE.md section 7). Every step can be rerun: API responses are
cached, finished work is skipped, and submitted batches are resumed rather than paid for twice.
`select_sample` is an addition to the documented list: the Wikidata adaptation pairs decide which
works to fetch, so the sample has to be chosen before the fetch steps.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import anthropic
import openai

from median import llm
from median.config import (
    CACHE_DIR,
    DUMPS_DIR,
    GOLD_ADAPTATIONS_PATH,
    GOLD_VIBES_PATH,
    SAMPLE_PLAN_PATH,
)
from median.db import init_db
from pipeline import catalog
from pipeline.enrich import embed as embed_step
from pipeline.enrich import index as index_step
from pipeline.enrich import link_check as link_step
from pipeline.enrich import match as match_step
from pipeline.enrich import vibe_cards
from pipeline.gold import read_gold_adaptations, read_gold_vibe_refs
from pipeline.report import build_report, check_targets, format_report
from pipeline.sample_plan import SamplePlan, SampleSpec, bare_id, select_sample
from pipeline.sample_seed import forced_from_entries, load_sample
from pipeline.sources import ol_dumps, openlibrary, tmdb, wikidata
from pipeline.sources.http import CachedClient

logger = logging.getLogger(__name__)

STEPS = (
    "select_sample",
    "fetch_tmdb",
    "load_openlibrary",
    "wikidata_links",
    "match",
    "link_check",
    "vibe_cards",
    "embed",
    "index",
    "report",
)
LLM_STEPS = ("link_check", "vibe_cards", "embed")  # the ones that cost money
SAMPLE_SPEC = SampleSpec()
FULL_SPEC = SampleSpec(
    movies=20_000,
    books=20_000,
    min_pairs=0,
    pair_buffer=1.0,
    min_description_chars=0,  # the documented rule: a description or subjects
    book_candidates=60_000,
)
DEFAULT_MAX_USD = 1.0
FIRST_FILM_YEAR = 1950


class BuildError(RuntimeError):
    """A step cannot continue (for example, an LLM step would cost more than --max-usd)."""


@dataclass
class BuildContext:
    conn: sqlite3.Connection
    sample: bool
    spec: SampleSpec
    max_usd: float = DEFAULT_MAX_USD
    dumps_dir: Path = DUMPS_DIR
    cache_dir: Path = CACHE_DIR
    plan_path: Path = SAMPLE_PLAN_PATH
    gold_path: Path = GOLD_ADAPTATIONS_PATH
    vibes_path: Path = GOLD_VIBES_PATH
    api: str = "batch"  # 'batch' (half price, queued) or 'regular' (full price, parallel, no queue)
    problems: list[str] = field(default_factory=list)

    @property
    def use_batch(self) -> bool:
        return self.api == "batch"

    @property
    def dispatch(self) -> Callable[..., list[llm.BatchResult]]:
        """How bulk LLM requests are sent: the Batches API, or many regular calls at once."""
        return llm.run_batch if self.use_batch else llm.run_parallel


# Client factories: module-level so tests can replace them -------------------------------------


def _tmdb_client() -> CachedClient:
    return tmdb.make_client()


def _openlibrary_client() -> CachedClient:
    return openlibrary.make_client()


def _wikidata_client() -> CachedClient:
    return wikidata.make_client()


def _guard(ctx: BuildContext, what: str, usd: float) -> None:
    """Log an LLM step's estimated cost and refuse to go on if it is above --max-usd."""
    logger.info("%s: estimated cost about $%.4f (limit --max-usd %.2f)", what, usd, ctx.max_usd)
    if usd > ctx.max_usd:
        raise BuildError(f"{what} would cost about ${usd:.4f}, above --max-usd {ctx.max_usd:.2f}")


def _dumps(ctx: BuildContext, *names: str) -> dict[str, Path]:
    return {n: ol_dumps.download_dump(n, dest_dir=ctx.dumps_dir) for n in names}


def _reading_counts(ctx: BuildContext, dumps: dict[str, Path]) -> dict[str, int]:
    return ol_dumps.top_reading_counts(
        dumps["reading-log"], ctx.cache_dir / "openlibrary" / "reading_top.json"
    )


def _works_cache(ctx: BuildContext) -> ol_dumps.RecordCache:
    return ol_dumps.RecordCache(
        ctx.cache_dir / "openlibrary" / "works.jsonl", ol_dumps.project_work
    )


def _authors_cache(ctx: BuildContext) -> ol_dumps.RecordCache:
    return ol_dumps.RecordCache(
        ctx.cache_dir / "openlibrary" / "authors.jsonl", ol_dumps.project_author
    )


# Steps -------------------------------------------------------------------------------------------


def step_select_sample(ctx: BuildContext) -> None:
    """Choose which movies and books to fetch and save the plan."""
    dumps = _dumps(ctx, "reading-log", "authors", "works")
    counts = _reading_counts(ctx, dumps)
    with _tmdb_client() as tmdb_client, _openlibrary_client() as ol_client:
        review = load_sample(ctx.conn, tmdb_client, ol_client)  # the 50 reviewed works, cached
    movies, books, pairs = forced_from_entries(review.entries)
    for pair in read_gold_adaptations(ctx.gold_path):
        movies.append(pair.movie_tmdb_id)
        books.append(pair.book_ol_id)
        pairs.append((pair.movie_tmdb_id, pair.book_ol_id))
    vibe_movies, vibe_books = read_gold_vibe_refs(ctx.vibes_path)  # the vibe eval needs these too
    movies += vibe_movies
    books += vibe_books
    with _wikidata_client() as wikidata_client:
        wikidata_pairs = wikidata.fetch_adaptation_pairs(wikidata_client)

    works_cache = _works_cache(ctx)

    def has_description(ids: list[str]) -> dict[str, bool]:
        records = works_cache.get_many([f"/works/{i}" for i in ids], dumps["works"])
        return {
            i: ol_dumps.has_description(records[f"/works/{i}"], ctx.spec.min_description_chars)
            for i in ids
        }

    with _tmdb_client() as tmdb_client:

        def discover(n: int) -> list[int]:
            if ctx.sample:
                return tmdb.discover_ids(tmdb_client, n)
            return tmdb.discover_sliced(
                tmdb_client, n, range(FIRST_FILM_YEAR, date.today().year + 1)
            )

        plan = select_sample(
            ctx.spec,
            forced_movies=list(dict.fromkeys(movies)),
            forced_books=[bare_id(b) for b in dict.fromkeys(books)],
            forced_pairs=list(dict.fromkeys(pairs)),
            wikidata_pairs=wikidata_pairs,
            reading_counts=counts,
            has_description=has_description,
            discover_movies=discover,
        )
    plan.save(ctx.plan_path)
    logger.info(
        "Plan: %d movies, %d books, %d adaptation pairs -> %s",
        len(plan.movies),
        len(plan.books),
        len(plan.pairs),
        ctx.plan_path,
    )


def step_fetch_tmdb(ctx: BuildContext) -> None:
    plan = SamplePlan.load(ctx.plan_path)
    with _tmdb_client() as client:
        report = catalog.load_tmdb_movies(ctx.conn, client, plan.movies)
    logger.info("TMDB: %d movies loaded, %d skipped", report.loaded, len(report.skipped))
    for line in report.skipped[:10]:
        logger.info("  skipped %s", line)


def step_load_openlibrary(ctx: BuildContext) -> None:
    plan = SamplePlan.load(ctx.plan_path)
    dumps = _dumps(ctx, "reading-log", "authors", "works")
    report = catalog.load_openlibrary_books(
        ctx.conn,
        plan.books,
        reading_counts=_reading_counts(ctx, dumps),
        works_cache=_works_cache(ctx),
        authors_cache=_authors_cache(ctx),
        works_dump=dumps["works"],
        authors_dump=dumps["authors"],
        min_description_chars=ctx.spec.min_description_chars,
    )
    logger.info("Open Library: %d books loaded, %d skipped", report.loaded, len(report.skipped))
    for line in report.skipped[:10]:
        logger.info("  skipped %s", line)


def step_wikidata_links(ctx: BuildContext) -> None:
    with _wikidata_client() as client:
        adaptations = wikidata.fetch_adaptation_pairs(client)
        series = wikidata.fetch_series_pairs(client)
    counts = catalog.link_wikidata(ctx.conn, adaptations, series)
    logger.info("Wikidata links in the catalog: %s", counts)


def step_match(ctx: BuildContext) -> None:
    with _tmdb_client() as client:
        match_step.match_links(ctx.conn, client)


def step_link_check(ctx: BuildContext) -> None:
    with _tmdb_client() as client:
        pairs, usd = link_step.estimate_link_check(ctx.conn, client, batch=ctx.use_batch)
        logger.info("link_check: %d pair(s) not yet judged", pairs)
        if pairs:
            _guard(ctx, "link_check", usd)
        link_step.check_links(ctx.conn, client, dispatch=ctx.dispatch)


def step_vibe_cards(ctx: BuildContext) -> None:
    template = vibe_cards.load_template()
    ids = [r["id"] for r in ctx.conn.execute("SELECT id FROM works ORDER BY id")]
    works = [
        vibe_cards.load_work(ctx.conn, i)
        for i in ids
        if vibe_cards.load_stored_card(ctx.conn, i) is None
    ]
    logger.info("vibe_cards: %d work(s) need a card", len(works))
    if not works:
        return
    est = vibe_cards.estimate(
        [vibe_cards.render_prompt(template, w) for w in works], batch=ctx.use_batch
    )
    _guard(ctx, "vibe_cards (with retries)", est.ceiling_usd)
    outcome = vibe_cards.generate_cards_batch(ctx.conn, works, template, dispatch=ctx.dispatch)
    for work, problems in outcome.failed[:20]:
        logger.warning(
            "  no card for %s (%s): %s", work.title, work.media_type, "; ".join(problems)
        )


def step_embed(ctx: BuildContext) -> None:
    rows = embed_step.pending_rows(ctx.conn)
    logger.info("embed: %d card(s) need a vector", len(rows))
    if rows:
        _guard(ctx, "embed", embed_step.estimate_usd(rows))
    embed_step.embed_cards(ctx.conn)


def step_index(ctx: BuildContext) -> None:
    index_step.build_index(ctx.conn)


def step_report(ctx: BuildContext) -> None:
    report = build_report(ctx.conn)
    for line in format_report(report):
        logger.info(line)
    ctx.problems = check_targets(report, ctx.spec, require_cards=report.cards > 0)
    for problem in ctx.problems:
        logger.warning("TARGET NOT MET: %s", problem)
    if not ctx.problems:
        logger.info("All catalog targets met.")


STEP_FUNCS: dict[str, Callable[[BuildContext], None]] = {
    "select_sample": step_select_sample,
    "fetch_tmdb": step_fetch_tmdb,
    "load_openlibrary": step_load_openlibrary,
    "wikidata_links": step_wikidata_links,
    "match": step_match,
    "link_check": step_link_check,
    "vibe_cards": step_vibe_cards,
    "embed": step_embed,
    "index": step_index,
    "report": step_report,
}


def select_steps(step: str | None, until: str | None) -> list[str]:
    """Return the steps to run: one step, everything up to `until`, or all of them."""
    if step:
        return [step]
    if until:
        return list(STEPS[: STEPS.index(until) + 1])
    return list(STEPS)


def run_steps(ctx: BuildContext, steps: list[str]) -> None:
    for name in steps:
        logger.info("== %s ==", name)
        STEP_FUNCS[name](ctx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/median.db.")
    parser.add_argument("--sample", action="store_true", help="build the ~2,000-work dev catalog")
    parser.add_argument("--step", choices=STEPS, help="run only this step")
    parser.add_argument("--until", choices=STEPS, help="run every step up to and including this")
    parser.add_argument("--yes", action="store_true", help="confirm a FULL build (about $5)")
    parser.add_argument(
        "--api",
        choices=("batch", "regular"),
        default="batch",
        help="batch: half price but queued, for any size (default); regular: full price, no queue",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=DEFAULT_MAX_USD,
        help="refuse any single LLM step estimated above this (default %(default)s)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.sample and not args.yes:
        logger.error(
            "A full build costs about $5 in API calls and downloads ~5 GB. Use --sample for the "
            "dev catalog (about $0.20), or add --yes if you really mean the full build."
        )
        return 2
    conn = init_db()
    ctx = BuildContext(
        conn=conn,
        sample=args.sample,
        spec=SAMPLE_SPEC if args.sample else FULL_SPEC,
        max_usd=args.max_usd,
        api=args.api,
    )
    try:
        run_steps(ctx, select_steps(args.step, args.until))
    except BuildError as exc:
        logger.error("Stopped: %s", exc)
        return 1
    except (anthropic.APIStatusError, openai.APIStatusError) as exc:
        # e.g. 401 bad key, or 'no credits'. Finished work is saved; fix the account and rerun.
        logger.error(
            "%s API error %s: %s", type(exc).__module__.split(".")[0], exc.status_code, exc.message
        )
        return 3
    finally:
        conn.close()
    return 1 if ctx.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
