"""Judge unverified adaptation links with a Haiku batch (PROMPTS.md section 2).

Keeps a link only when the model says it is an adaptation with confidence >= 0.7 (then
`verified=1`); otherwise the link is deleted. Verdicts are cached on disk by pair, so a rejected
link that `match` proposes again next run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from median import llm
from median.config import AGENT_MODEL, CACHE_DIR, LINK_PROMPT_VERSION
from pipeline.enrich.prompting import fill, load_prompt
from pipeline.enrich.vibe_cards import estimate
from pipeline.sources.http import CachedClient
from pipeline.sources.tmdb import crew_names, raw_movie

logger = logging.getLogger(__name__)

PURPOSE = "link_check"
KEEP_CONFIDENCE = 0.7
WRITER_JOBS = {"Writer", "Screenplay", "Novel", "Book", "Author", "Story"}
MAX_TOKENS = 150
EXPECTED_OUTPUT_TOKENS = 60
CACHE_PATH = CACHE_DIR / "link_check" / f"{LINK_PROMPT_VERSION}.json"


class LinkVerdict(BaseModel):
    is_adaptation: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""

    @property
    def keep(self) -> bool:
        return self.is_adaptation and self.confidence >= KEEP_CONFIDENCE


@dataclass
class LinkCheckReport:
    kept: int = 0
    dropped: int = 0
    unknown: list[str] = field(default_factory=list)  # pairs with no usable verdict this run


def parse_verdict(text: str) -> LinkVerdict:
    """Parse the model's reply (tolerating code fences). Raises ValueError if invalid."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return LinkVerdict.model_validate_json(text[start : end + 1])


def _load_cache(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class _Pending:
    from_work: int
    to_work: int
    key: str  # 'tmdb:<id>|ol:<id>', stable across rebuilds
    prompt: str


def _pending(conn: sqlite3.Connection, tmdb: CachedClient, template: str) -> list[_Pending]:
    rows = conn.execute(
        "SELECT l.from_work, l.to_work, f.title AS ft, f.year AS fy, f.synopsis AS fs, "
        "b.title AS bt, b.year AS by, b.creators AS bc, b.synopsis AS bs, "
        "(SELECT external_id FROM external_ids WHERE work_id = f.id AND source = 'tmdb') AS tmdb, "
        "(SELECT external_id FROM external_ids WHERE work_id = b.id AND source = 'openlibrary') "
        "AS ol FROM work_links l JOIN works f ON f.id = l.from_work "
        "JOIN works b ON b.id = l.to_work "
        "WHERE l.link_type = 'adaptation_of' AND l.verified = 0 ORDER BY l.from_work, l.to_work"
    ).fetchall()
    pending = []
    for r in rows:
        writers = crew_names(raw_movie(tmdb, r["tmdb"]), WRITER_JOBS)
        prompt = fill(
            template,
            {
                "film_title": r["ft"],
                "film_year": str(r["fy"] or "unknown"),
                "film_writers": ", ".join(writers) or "unknown",
                "film_synopsis": r["fs"] or "unknown",
                "book_title": r["bt"],
                "book_year": str(r["by"] or "unknown"),
                "book_author": ", ".join(json.loads(r["bc"] or "[]")) or "unknown",
                "book_synopsis": r["bs"] or "unknown",
            },
        )
        pending.append(
            _Pending(r["from_work"], r["to_work"], f"tmdb:{r['tmdb']}|ol:{r['ol']}", prompt)
        )
    return pending


def estimate_link_check(
    conn: sqlite3.Connection, tmdb: CachedClient, cache_path: Path | None = None
) -> tuple[int, float]:
    """Return (uncached pairs to check, estimated batch cost in USD)."""
    cache = _load_cache(cache_path or CACHE_PATH)
    todo = [p for p in _pending(conn, tmdb, load_prompt("link_check")) if p.key not in cache]
    cost = estimate(
        [p.prompt for p in todo], output_tokens_each=EXPECTED_OUTPUT_TOKENS, batch=True
    ).usd
    return len(todo), cost


def check_links(
    conn: sqlite3.Connection,
    tmdb: CachedClient,
    *,
    model: str = AGENT_MODEL,
    cache_path: Path | None = None,
    poll_seconds: float = llm.BATCH_POLL_SECONDS,
    sleep: Callable[[float], None] = llm.sleep_for,
) -> LinkCheckReport:
    """Verify or drop every unverified adaptation link, calling the batch API only for new pairs."""
    cache_path = cache_path or CACHE_PATH
    cache = _load_cache(cache_path)
    pending = _pending(conn, tmdb, load_prompt("link_check"))
    todo = [p for p in pending if p.key not in cache]
    report = LinkCheckReport()

    if todo:
        requests = [
            llm.BatchRequest(
                f"l{p.from_work}_{p.to_work}",
                [{"role": "user", "content": p.prompt}],
                max_tokens=MAX_TOKENS,
            )
            for p in todo
        ]
        digest = hashlib.sha1("|".join(sorted(r.custom_id for r in requests)).encode()).hexdigest()
        cost = estimate(
            [p.prompt for p in todo], output_tokens_each=EXPECTED_OUTPUT_TOKENS, batch=True
        ).usd
        results = llm.run_batch(
            conn,
            purpose=PURPOSE,
            tag=f"{LINK_PROMPT_VERSION}-{digest[:10]}",
            model=model,
            requests=requests,
            estimated_usd=cost,
            poll_seconds=poll_seconds,
            sleep=sleep,
        )
        by_id = {r.custom_id: r for r in results}
        for p in todo:
            result = by_id.get(f"l{p.from_work}_{p.to_work}")
            if result is None or result.status != "succeeded":
                continue
            try:
                cache[p.key] = parse_verdict(result.text or "").model_dump()
            except ValueError:
                logger.warning("link_check: unusable reply for %s", p.key)
        _save_cache(cache_path, cache)

    for p in pending:
        verdict = cache.get(p.key)
        if verdict is None:
            report.unknown.append(p.key)
            continue
        parsed = LinkVerdict(**verdict)
        if parsed.keep:
            conn.execute(
                "UPDATE work_links SET verified = 1, confidence = ? "
                "WHERE from_work = ? AND to_work = ? AND link_type = 'adaptation_of'",
                (parsed.confidence, p.from_work, p.to_work),
            )
            report.kept += 1
        else:
            conn.execute(
                "DELETE FROM work_links WHERE from_work = ? AND to_work = ? "
                "AND link_type = 'adaptation_of'",
                (p.from_work, p.to_work),
            )
            report.dropped += 1
    conn.commit()
    logger.info(
        "link_check: kept %d, dropped %d, no verdict for %d",
        report.kept,
        report.dropped,
        len(report.unknown),
    )
    return report
