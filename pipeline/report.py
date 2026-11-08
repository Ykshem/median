"""A health report for the built catalog: counts, link coverage, cards, vectors and spend."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pipeline.enrich.vibe_cards import check_rules, load_stored_card, load_work
from pipeline.sample_plan import SampleSpec

COVERAGE_TOLERANCE = 0.95  # a few works lose their synopsis between planning and loading


@dataclass(frozen=True)
class Report:
    works: dict[str, int]
    cards: int
    missing_cards: int
    flagged_cards: int  # stored cards that break a PROMPTS.md rule
    vectors: int
    links: list[tuple[str, str, int, int]]  # (link_type, source, verified, count)
    adaptation_pairs: int  # verified adaptation_of links
    unverified_links: int
    spend: dict[str, float]  # USD by llm_usage.purpose


def build_report(conn: sqlite3.Connection) -> Report:
    """Collect the numbers. Reads the whole catalog, so it takes a second or two at 2,000 works."""
    works = dict(conn.execute("SELECT media_type, COUNT(*) FROM works GROUP BY media_type"))
    work_ids = [r[0] for r in conn.execute("SELECT id FROM works ORDER BY id")]
    flagged = 0
    cards = 0
    for work_id in work_ids:
        card = load_stored_card(conn, work_id)
        if card is None:
            continue
        cards += 1
        flagged += bool(check_rules(card, load_work(conn, work_id)).blocking)
    links = [
        (r[0], r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT link_type, source, verified, COUNT(*) FROM work_links "
            "GROUP BY link_type, source, verified ORDER BY link_type, source, verified"
        )
    ]
    return Report(
        works=works,
        cards=cards,
        missing_cards=len(work_ids) - cards,
        flagged_cards=flagged,
        vectors=conn.execute("SELECT COUNT(*) FROM vec_vibes").fetchone()[0],
        links=links,
        adaptation_pairs=conn.execute(
            "SELECT COUNT(*) FROM work_links WHERE link_type = 'adaptation_of' AND verified = 1"
        ).fetchone()[0],
        unverified_links=conn.execute(
            "SELECT COUNT(*) FROM work_links WHERE verified = 0"
        ).fetchone()[0],
        spend={
            r[0] or "unknown": r[1]
            for r in conn.execute(
                "SELECT purpose, ROUND(SUM(cost_usd), 4) FROM llm_usage GROUP BY purpose"
            )
        },
    )


def format_report(report: Report) -> list[str]:
    """Return the report as plain lines for logging."""
    total_works = sum(report.works.values())
    lines = [
        "== Catalog report ==",
        "Works: "
        + (", ".join(f"{n} {kind}s" for kind, n in sorted(report.works.items())) or "none"),
        f"Vibe cards: {report.cards} / {total_works} "
        f"({report.missing_cards} missing, {report.flagged_cards} break a rule)",
        f"Vectors: {report.vectors}",
        f"Verified adaptation pairs: {report.adaptation_pairs} "
        f"(unverified links waiting for link_check: {report.unverified_links})",
        "Links by type / source / verified:",
    ]
    lines += [f"  {t:15} {s:12} verified={v}  {n}" for t, s, v, n in report.links] or ["  (none)"]
    if report.spend:
        parts = ", ".join(f"{purpose} ${usd:.4f}" for purpose, usd in sorted(report.spend.items()))
        lines.append(f"Spend: ${sum(report.spend.values()):.4f} ({parts})")
    else:
        lines.append("Spend: $0")
    return lines


def check_targets(report: Report, spec: SampleSpec, *, require_cards: bool) -> list[str]:
    """Return the ways the catalog falls short of the spec (empty when it meets every target)."""
    problems = []
    for media, wanted in (("movie", spec.movies), ("book", spec.books)):
        have = report.works.get(media, 0)
        if have < wanted * COVERAGE_TOLERANCE:
            problems.append(f"only {have} {media}s (wanted about {wanted})")
    if report.adaptation_pairs < spec.min_pairs:
        problems.append(
            f"only {report.adaptation_pairs} verified adaptation pairs (need {spec.min_pairs})"
        )
    if require_cards:
        if report.missing_cards:
            problems.append(f"{report.missing_cards} works have no vibe card")
        total = sum(report.works.values())
        if report.vectors != total:
            problems.append(f"{report.vectors} vectors for {total} works")
    return problems
