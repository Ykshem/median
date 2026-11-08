"""Generate vibe cards (ARCHITECTURE.md section 4) and a team-readable review file.

This is the regular-API path used for the 50-work review set. The full catalog build will use the
Message Batches API instead (half price). Every call goes through `median.llm`, so the daily budget
guard and `llm_usage` logging apply.

CLI (all steps are idempotent; responses are cached and finished cards are skipped):
    uv run python -m pipeline.enrich.vibe_cards --fetch      # TMDB + Open Library only (free)
    uv run python -m pipeline.enrich.vibe_cards --estimate   # print the LLM cost, make no LLM call
    uv run python -m pipeline.enrich.vibe_cards --run        # generate cards, write the review file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import anthropic
from pydantic import ValidationError

from median import llm
from median.config import AGENT_MODEL, REVIEW_PATH, VIBE_PROMPT_VERSION
from median.db import init_db
from median.models import VibeCard, Work
from pipeline.enrich.prompting import fill, load_prompt
from pipeline.sample_seed import SampleEntry, load_sample
from pipeline.sources import normalize_title
from pipeline.sources import openlibrary as openlibrary_source
from pipeline.sources import tmdb as tmdb_source

logger = logging.getLogger(__name__)

PURPOSE = "vibe_card"
MAX_CALLS_PER_WORK = 2  # the first try plus one retry
EXPECTED_OUTPUT_TOKENS = 250  # one JSON card; deliberately on the high side
CHARS_PER_TOKEN = 4  # rough rule of thumb, good enough for an estimate
DEFAULT_MAX_USD = 0.25
MAX_CARD_TOKENS = 600
SYNOPSIS_PREVIEW_CHARS = 400

# The words PROMPTS.md forbids, because they describe the medium rather than the story.
MEDIUM_WORDS = re.compile(
    r"\b(films?|movies?|novels?|books?|readers?|viewers?|chapters?|pages|cinematography|"
    r"prose|runtime)\b",
    re.IGNORECASE,
)
MIN_TITLE_LEAK_CHARS = 10  # shorter titles ("Get Out") are ordinary phrases: warn, don't retry


# Prompt ------------------------------------------------------------------------------------------


def load_template() -> str:
    """Return the vibe-card prompt (median/prompts/vibe_card.md)."""
    return load_prompt("vibe_card")


def render_prompt(template: str, work: Work) -> str:
    """Fill the template's `{placeholders}` with a work's data in a single pass.

    Single pass (not str.format or repeated replace) means braces or placeholder-looking text inside
    a synopsis are never expanded again. Catalog text is untrusted data (AGENTS.md).
    """
    return fill(
        template,
        {
            "media_type": work.media_type,
            "title": work.title,
            "year": str(work.year) if work.year else "unknown",
            "synopsis": work.synopsis or "unknown",
            "subjects": ", ".join(work.subjects) or "unknown",
        },
    )


# Parsing and rules -------------------------------------------------------------------------------


def parse_card(text: str) -> VibeCard:
    """Parse the model's reply into a VibeCard, tolerating code fences or chatter around the JSON.

    Raises ValueError (pydantic.ValidationError is one) if there is no valid card.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return VibeCard.model_validate_json(text[start : end + 1])


@dataclass(frozen=True)
class RuleReport:
    blocking: list[str] = field(default_factory=list)  # break a PROMPTS.md rule: retry, don't store
    warnings: list[str] = field(default_factory=list)  # probably fine; show to the reviewers


def check_rules(card: VibeCard, work: Work) -> RuleReport:
    """Check a card against the PROMPTS.md rules: no medium words, titles or creator names."""
    text = card.to_text()
    padded = f" {normalize_title(text)} "
    blocking: list[str] = []
    warnings: list[str] = []

    medium = sorted({m.group(0).lower() for m in MEDIUM_WORDS.finditer(text)})
    if medium:
        blocking.append(f"mentions the medium ({', '.join(medium)})")

    title = normalize_title(work.title)
    if title and f" {title} " in padded:
        if len(title) >= MIN_TITLE_LEAK_CHARS and len(title.split()) >= 2:
            blocking.append(f"contains the title {work.title!r}")
        else:
            warnings.append(f"contains the title {work.title!r} (may be an ordinary word)")

    for creator in work.creators:
        full = normalize_title(creator)
        if full and f" {full} " in padded:
            blocking.append(f"contains the creator name {creator!r}")
            continue
        surname = full.split()[-1] if full else ""
        if len(surname) >= 4 and f" {surname} " in padded:
            warnings.append(f"contains {surname!r}, the surname of {creator} (may be coincidence)")
    return RuleReport(blocking, warnings)


# Generation --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CardResult:
    card: VibeCard | None
    report: RuleReport
    calls: int

    @property
    def ok(self) -> bool:
        return self.card is not None and not self.report.blocking


def _problem_summary(exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc'])) or 'card'}: {e['msg']}" for e in exc.errors()]
        return "invalid card JSON (" + "; ".join(parts)[:300] + ")"
    return f"invalid card JSON ({exc})"


def _reply_text(response: object) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def generate_card(
    conn: sqlite3.Connection, work: Work, template: str, *, model: str = AGENT_MODEL
) -> CardResult:
    """Ask the model for one vibe card, retrying once if it is invalid or breaks a rule.

    Raises llm.BudgetExceeded if the daily limit is reached. A card that still fails after the
    retry is returned (when it parsed) with its problems, so reviewers can see real failures.
    """
    prompt = render_prompt(template, work)
    card: VibeCard | None = None
    report = RuleReport()
    for call in range(1, MAX_CALLS_PER_WORK + 1):
        response = llm.complete(
            conn,
            purpose=PURPOSE,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_CARD_TOKENS,
        )
        try:
            card = parse_card(_reply_text(response))
        except ValueError as exc:
            card, report = None, RuleReport(blocking=[_problem_summary(exc)])
        else:
            report = check_rules(card, work)
        if not report.blocking:
            return CardResult(card, report, call)
        logger.info("  %s: attempt %d rejected: %s", work.title, call, "; ".join(report.blocking))
        prompt = (
            render_prompt(template, work)
            + "\n\nYour previous answer was rejected: "
            + "; ".join(report.blocking)
            + ". Return corrected JSON only."
        )
    return CardResult(card, report, MAX_CALLS_PER_WORK)


# Storage -----------------------------------------------------------------------------------------


def load_work(conn: sqlite3.Connection, work_id: int) -> Work:
    """Return a Work built from its `works` row."""
    row = conn.execute("SELECT * FROM works WHERE id = ?", (work_id,)).fetchone()
    return Work(
        id=row["id"],
        media_type=row["media_type"],
        title=row["title"],
        year=row["year"],
        creators=json.loads(row["creators"] or "[]"),
        synopsis=row["synopsis"],
        subjects=json.loads(row["subjects"] or "[]"),
        popularity=row["popularity"],
        cover_url=row["cover_url"],
    )


def load_stored_card(conn: sqlite3.Connection, work_id: int) -> VibeCard | None:
    """Return the stored card for the current prompt version, or None."""
    row = conn.execute(
        "SELECT card_json FROM vibe_cards WHERE work_id = ? AND prompt_version = ?",
        (work_id, VIBE_PROMPT_VERSION),
    ).fetchone()
    return VibeCard.model_validate_json(row["card_json"]) if row else None


def store_card(conn: sqlite3.Connection, work_id: int, card: VibeCard, model: str) -> None:
    """Insert or replace a work's card (with the text that will be embedded later)."""
    conn.execute(
        "INSERT OR REPLACE INTO vibe_cards (work_id, card_json, card_text, model, prompt_version) "
        "VALUES (?, ?, ?, ?, ?)",
        (work_id, card.model_dump_json(), card.to_text(), model, VIBE_PROMPT_VERSION),
    )
    conn.commit()


def vibe_spend(conn: sqlite3.Connection) -> float:
    """Return all USD ever logged for vibe-card calls."""
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM llm_usage WHERE purpose = ?", (PURPOSE,)
    ).fetchone()
    return float(row["total"])


# Cost estimate -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    ceiling_usd: float  # if every card needed its retry


def estimate(
    prompts: list[str],
    model: str = AGENT_MODEL,
    *,
    output_tokens_each: int = EXPECTED_OUTPUT_TOKENS,
    batch: bool = False,
) -> Estimate:
    """Estimate the cost of one reply per prompt (about 4 characters per token)."""
    input_tokens = sum(math.ceil(len(p) / CHARS_PER_TOKEN) for p in prompts)
    output_tokens = output_tokens_each * len(prompts)
    usd = llm.cost_usd(model, input_tokens, output_tokens, batch=batch)
    return Estimate(len(prompts), input_tokens, output_tokens, usd, usd * MAX_CALLS_PER_WORK)


def pending_prompts(
    conn: sqlite3.Connection, entries: list[SampleEntry], template: str
) -> list[str]:
    """Return the rendered prompts for works that don't yet have a card for this prompt version."""
    return [
        render_prompt(template, load_work(conn, e.work_id))
        for e in entries
        if load_stored_card(conn, e.work_id) is None
    ]


# Batch generation (bulk, half price) -------------------------------------------------------------


@dataclass
class BatchOutcome:
    stored: int = 0
    failed: list[tuple[Work, list[str]]] = field(default_factory=list)  # still bad after the retry


def _round_tag(round_no: int, chunk: int, ids: list[int]) -> str:
    digest = hashlib.sha1(",".join(map(str, sorted(ids))).encode()).hexdigest()[:10]
    return f"{VIBE_PROMPT_VERSION}-r{round_no}-c{chunk}-{digest}"


def generate_cards_batch(
    conn: sqlite3.Connection,
    works: list[Work],
    template: str,
    *,
    model: str = AGENT_MODEL,
    poll_seconds: float = llm.BATCH_POLL_SECONDS,
    sleep: Callable[[float], None] = llm.sleep_for,
) -> BatchOutcome:
    """Generate and store cards for works that lack one, using the Batches API.

    Round 1 sends every card; round 2 resends only the invalid ones with the reason appended
    (a single retry, as for the regular path). Good cards are committed as they are read, and a
    rerun resumes any batch already submitted. Cards still bad after round 2 are returned, not
    stored.
    """
    pending = [w for w in works if load_stored_card(conn, w.id) is None]
    outcome = BatchOutcome()
    feedback: dict[int, str] = {}
    problems: dict[int, list[str]] = {}
    for round_no in (1, 2):
        if not pending:
            break
        failed: list[Work] = []
        for start in range(0, len(pending), llm.MAX_BATCH_REQUESTS):
            chunk = pending[start : start + llm.MAX_BATCH_REQUESTS]
            prompts = {w.id: render_prompt(template, w) + feedback.get(w.id, "") for w in chunk}
            requests = [
                llm.BatchRequest(
                    f"w{w.id}",
                    [{"role": "user", "content": prompts[w.id]}],
                    max_tokens=MAX_CARD_TOKENS,
                )
                for w in chunk
            ]
            results = llm.run_batch(
                conn,
                purpose=PURPOSE,
                tag=_round_tag(round_no, start // llm.MAX_BATCH_REQUESTS, [w.id for w in chunk]),
                model=model,
                requests=requests,
                estimated_usd=estimate(list(prompts.values()), model, batch=True).usd,
                poll_seconds=poll_seconds,
                sleep=sleep,
            )
            by_id = {r.custom_id: r for r in results}
            for work in chunk:
                result = by_id.get(f"w{work.id}")
                if result is None or result.status != "succeeded":
                    status = result.status if result else "missing"
                    problems[work.id] = [f"batch request {status}"]
                    failed.append(work)
                    continue
                try:
                    card = parse_card(result.text or "")
                except ValueError as exc:
                    problems[work.id] = [_problem_summary(exc)]
                    failed.append(work)
                    continue
                report = check_rules(card, work)
                if report.blocking:
                    problems[work.id] = report.blocking
                    failed.append(work)
                    continue
                store_card(conn, work.id, card, model)
                outcome.stored += 1
        for work in failed:
            feedback[work.id] = (
                "\n\nYour previous answer was rejected: "
                + "; ".join(problems[work.id])
                + ". Return corrected JSON only."
            )
        pending = failed
    outcome.failed = [(w, problems[w.id]) for w in pending]
    logger.info("vibe cards (batch): %d stored, %d failed", outcome.stored, len(outcome.failed))
    return outcome


# Run ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewRow:
    entry: SampleEntry
    work: Work
    card: VibeCard | None
    report: RuleReport
    cached: bool  # True when the card already existed (no API call this run)

    @property
    def needs_attention(self) -> bool:
        return self.card is None or bool(self.report.blocking)


def run(
    conn: sqlite3.Connection,
    entries: list[SampleEntry],
    *,
    model: str = AGENT_MODEL,
    template: str | None = None,
) -> tuple[list[ReviewRow], bool]:
    """Generate and store a card for every entry that lacks one; return (rows, hit_budget_limit).

    Each good card is committed straight away, so an interrupted run resumes where it stopped.
    Failed cards are not stored, but are returned for the review file.
    """
    template = template or load_template()
    rows: list[ReviewRow] = []
    for i, entry in enumerate(entries, 1):
        work = load_work(conn, entry.work_id)
        stored = load_stored_card(conn, entry.work_id)
        if stored is not None:
            rows.append(ReviewRow(entry, work, stored, check_rules(stored, work), cached=True))
            continue
        try:
            result = generate_card(conn, work, template, model=model)
        except llm.BudgetExceeded as exc:
            logger.error("Stopping: %s", exc)
            return rows, True
        if result.ok:
            store_card(conn, entry.work_id, result.card, model)
        rows.append(ReviewRow(entry, work, result.card, result.report, cached=False))
        logger.info(
            "[%d/%d] %s -> %s",
            i,
            len(entries),
            work.title,
            "ok" if result.ok else "NEEDS ATTENTION",
        )
    return rows, False


# Review file -------------------------------------------------------------------------------------

CHECKLIST = """\
## How to review

For each card, ask:

1. **No medium words.** It never says film, movie, novel, book, reader, viewer, chapter, pages,
   prose.
2. **No spoilers**, and no titles, character names or creator names.
3. **Plain, shared vocabulary.** Themes and moods use common words ("grief", "cozy", "eerie") that
   other cards would also use.
4. **Pairs read as alike.** In the adaptation pairs below, the movie's card and the book's card
   should feel like the same story. If not, the template is losing the signal we need.
5. **Honest.** It says `unknown` instead of guessing when the source text is thin.

Write notes under each card. Anything marked ⚠️ broke a rule or looked suspicious."""


def _entry_section(row: ReviewRow, level: str) -> str:
    work = row.work
    kind = "Movie" if work.media_type == "movie" else "Book"
    year = f" ({work.year})" if work.year else ""
    by = f" · {', '.join(work.creators)}" if work.creators else ""
    synopsis = (work.synopsis or "").strip()
    if len(synopsis) > SYNOPSIS_PREVIEW_CHARS:
        synopsis = synopsis[:SYNOPSIS_PREVIEW_CHARS].rstrip() + "…"
    lines = [f"{level} {kind}: {work.title}{year}{by}", "", f"> {synopsis}", ""]
    card = row.card
    if card is not None:
        lines += [
            f"- **Logline:** {card.logline}",
            f"- **Themes:** {', '.join(card.themes)}",
            f"- **Mood:** {', '.join(card.mood)}",
            f"- **Tone:** {card.tone}",
            f"- **Setting:** {card.setting}",
            f"- **Pacing / intensity / audience:** "
            f"{card.pacing} / {card.intensity} / {card.audience}",
            f"- **Hooks:** {', '.join(card.hooks)}",
            "",
        ]
    else:
        lines += ["_No valid card was produced._", ""]
    for problem in row.report.blocking:
        lines.append(f"⚠️ **Rule broken:** {problem}")
    for warning in row.report.warnings:
        lines.append(f"⚠️ Check: {warning}")
    if row.report.blocking or row.report.warnings:
        lines.append("")
    lines += ["**Reviewer notes:** ", ""]
    return "\n".join(lines)


def render_review_md(
    rows: list[ReviewRow], *, model: str, cost_usd: float, today: date | None = None
) -> str:
    """Render the review file: header and checklist, adaptation pairs first, then the rest."""
    today = today or datetime.now(UTC).date()
    flagged = sum(r.needs_attention for r in rows)
    out = [
        f"# Vibe card review — prompt {VIBE_PROMPT_VERSION}",
        "",
        f"- Model: `{model}` · Generated: {today.isoformat()} · Cost of vibe-card calls: "
        f"${cost_usd:.4f}",
        f"- Works: {len(rows)} · need attention: {flagged}",
        "",
        CHECKLIST,
        "",
        "## Adaptation pairs",
        "",
    ]
    pairs: dict[str, list[ReviewRow]] = {}
    for row in rows:
        if row.entry.pair:
            pairs.setdefault(row.entry.pair, []).append(row)
    for n, (name, members) in enumerate(pairs.items(), 1):
        out += [f"### Pair {n} — {name}", ""]
        out += [
            _entry_section(r, "####")
            for r in sorted(members, key=lambda r: r.work.media_type != "movie")
        ]
    for media, heading in (("movie", "Movies"), ("book", "Books")):
        singles = [r for r in rows if not r.entry.pair and r.work.media_type == media]
        out += [f"## {heading}", ""]
        out += [_entry_section(r, "###") for r in singles]
    return "\n".join(out).rstrip() + "\n"


# CLI ---------------------------------------------------------------------------------------------


def _log_estimate(est: Estimate) -> None:
    logger.info(
        "Estimate: %d call(s), ~%s input + ~%s output tokens -> ~$%.4f "
        "(ceiling if every card is retried: $%.4f)",
        est.calls,
        f"{est.input_tokens:,}",
        f"{est.output_tokens:,}",
        est.usd,
        est.ceiling_usd,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the vibe-card review set.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true", help="fetch works only (no LLM calls)")
    mode.add_argument("--estimate", action="store_true", help="print the LLM cost, make no call")
    mode.add_argument("--run", action="store_true", help="generate cards and write the review file")
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD, help="abort above this")
    parser.add_argument("--out", type=Path, default=REVIEW_PATH, help="review file to write")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    conn = init_db()
    try:
        with tmdb_source.make_client() as tmdb, openlibrary_source.make_client() as ol:
            sample = load_sample(conn, tmdb, ol)
        logger.info("%d works loaded, %d skipped", len(sample.entries), len(sample.failures))
        if args.fetch:
            return 0

        template = load_template()
        est = estimate(pending_prompts(conn, sample.entries, template))
        _log_estimate(est)
        if args.estimate:
            return 0
        if est.ceiling_usd > args.max_usd:
            logger.error(
                "Ceiling $%.4f is above --max-usd %.2f; not running.", est.ceiling_usd, args.max_usd
            )
            return 2

        before = vibe_spend(conn)
        try:
            rows, hit_limit = run(conn, sample.entries, template=template)
        except anthropic.APIStatusError as exc:
            # e.g. 401 bad key, 400 "credit balance is too low". Nothing was stored for this
            # call and finished cards are kept, so fix the account and rerun --run.
            logger.error("Anthropic API error %s: %s", exc.status_code, exc.message)
            return 3
        spent = vibe_spend(conn) - before
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            render_review_md(rows, model=AGENT_MODEL, cost_usd=spent), encoding="utf-8"
        )
        flagged = sum(r.needs_attention for r in rows)
        logger.info(
            "Wrote %s: %d works, %d need attention, spent $%.4f this run (estimate was $%.4f)",
            args.out,
            len(rows),
            flagged,
            spent,
            est.usd,
        )
        return 1 if hit_limit else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
