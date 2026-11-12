"""Decide whether a book is fiction. Median's catalog holds only fiction books (ADR 0002).

Open Library has no fiction flag, only free-text subjects, and they are unreliable both ways:
*The Hobbit* is tagged "fantasy" (no "fiction"), *Harry Potter* "witches, monsters", while self-help
books are tagged "psychology", "success" and sometimes "nonfiction". So this works in two stages:

1. `classify_subjects` decides the clear cases from subjects alone, for free.
2. Everything else goes to a Haiku batch (`median/prompts/fiction_check.md`), whose verdicts are
   cached on disk by work id so a rerun costs nothing.

"Fiction" here means invented narrative: novels, novellas, short stories, plays, fairy tales and
myths told as stories, graphic-novel storytelling. Factual works (biography, memoir, history,
science, self-help, business, essays, criticism, textbooks, reference, religion) and poetry
collections are out. A verdict the model is unsure about counts as "not fiction".
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from median import llm
from median.config import AGENT_MODEL, CACHE_DIR, FICTION_PROMPT_VERSION
from pipeline.enrich.prompting import fill, load_prompt
from pipeline.enrich.vibe_cards import estimate

logger = logging.getLogger(__name__)

PURPOSE = "fiction_check"
KEEP_CONFIDENCE = 0.7
MAX_TOKENS = 200
EXPECTED_OUTPUT_TOKENS = 40
MAX_SYNOPSIS_CHARS = 700
CACHE_PATH = CACHE_DIR / "fiction" / f"{FICTION_PROMPT_VERSION}.json"

# A subject containing one of these is evidence of fiction ("Fiction", "Fantasy fiction",
# "Biographical fiction", "Science fiction").
FICTION_WORD = "fiction"
NONFICTION_WORDS = ("nonfiction", "non-fiction", "non fiction")
# "Fiction -- Technique", "Fiction -- History and criticism": books *about* fiction, not fiction.
ABOUT_FICTION = (
    "authorship",
    "technique",
    "history and criticism",
    "criticism and interpretation",
    "study and teaching",
    "stories, plots",
)
# Subjects that, as a whole, name a fiction genre. Exact match only: "romance" also occurs inside
# "romance languages", "mystery" inside "mystery of the universe".
FICTION_GENRES = frozenset(
    {
        "fantasy",
        "horror",
        "thriller",
        "thrillers",
        "novel",
        "novels",
        "short stories",
        "dystopias",
        "detective and mystery stories",
        "fairy tales",
        "ghost stories",
    }
)
# Subjects that mark a factual book. Matched as substrings of subjects that don't mention fiction.
MIN_MARKERS = 2  # distinct factual markers needed to call a book nonfiction without the word
NONFICTION_MARKERS = (
    "biograph",  # biography, autobiography, biographies
    "memoir",
    "self-help",
    "self help",
    "self-improvement",
    "self improvement",
    "personal development",
    "self-actualization",
    "personal finance",
    "investment",
    "textbook",
    "psychology",
    "conduct of life",
    "business",
)


def _norm(subjects: Iterable[str]) -> list[str]:
    return [s.strip().lower() for s in subjects if isinstance(s, str) and s.strip()]


def classify_subjects(subjects: Iterable[str]) -> bool | None:
    """Return True (fiction) or False (not) when the subjects are clear, else None (ask the model).

    Fiction evidence is any subject mentioning "fiction" (but not "nonfiction") or naming a fiction
    genre. Nonfiction evidence is an explicit "nonfiction" subject, or at least `MIN_MARKERS`
    different factual markers (biography, self-help, psychology, ...) in subjects that don't mention
    fiction, so "Biographical fiction" counts as fiction, not biography. One marker alone is not
    enough: "Conduct of life" is on plenty of children's novels. Evidence on both sides, or too
    little on either, is undecided.
    """
    fiction = explicit = False
    markers: set[str] = set()
    for subject in _norm(subjects):
        if any(w in subject for w in NONFICTION_WORDS):
            explicit = True
        elif (FICTION_WORD in subject and not any(a in subject for a in ABOUT_FICTION)) or (
            subject in FICTION_GENRES
        ):
            fiction = True
        else:
            markers.update(m for m in NONFICTION_MARKERS if m in subject)
    nonfiction = explicit or len(markers) >= MIN_MARKERS
    if fiction and nonfiction:
        return None
    if fiction:
        return True
    return False if nonfiction else None


class FictionVerdict(BaseModel):
    is_fiction: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""

    @property
    def keep(self) -> bool:
        return self.is_fiction and self.confidence >= KEEP_CONFIDENCE


def parse_verdict(text: str) -> FictionVerdict:
    """Parse the model's reply (tolerating code fences). Raises ValueError if invalid."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return FictionVerdict.model_validate_json(text[start : end + 1])


@dataclass(frozen=True)
class BookInfo:
    """What the classifier needs to know about a book (all from the Open Library dump)."""

    ol_id: str
    title: str
    year: int | None
    synopsis: str
    subjects: list[str]


@dataclass
class FictionReport:
    by_rules: int = 0
    by_model: int = 0
    undecided: list[str] = field(default_factory=list)  # no rule fired and no usable verdict


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def _prompt(template: str, book: BookInfo) -> str:
    return fill(
        template,
        {
            "title": book.title,
            "year": str(book.year or "unknown"),
            "synopsis": (book.synopsis or "unknown")[:MAX_SYNOPSIS_CHARS],
            "subjects": ", ".join(book.subjects[:15]) or "unknown",
        },
    )


def books_needing_model(books: Iterable[BookInfo], cache: Mapping[str, Any]) -> list[BookInfo]:
    """Return the books the rules can't decide and the cache has no verdict for."""
    return [b for b in books if classify_subjects(b.subjects) is None and b.ol_id not in cache]


def estimate_fiction_check(
    books: Iterable[BookInfo], cache_path: Path | None = None, *, batch: bool = True
) -> tuple[int, float]:
    """Return (books that would be sent to the model, estimated cost in USD)."""
    todo = books_needing_model(books, _load_cache(cache_path or CACHE_PATH))
    template = load_prompt("fiction_check")
    cost = estimate(
        [_prompt(template, b) for b in todo], output_tokens_each=EXPECTED_OUTPUT_TOKENS, batch=batch
    ).usd
    return len(todo), cost


def classify_books(
    conn: sqlite3.Connection,
    books: list[BookInfo],
    *,
    model: str = AGENT_MODEL,
    cache_path: Path | None = None,
    poll_seconds: float = llm.BATCH_POLL_SECONDS,
    sleep: Callable[[float], None] = llm.sleep_for,
    dispatch: Callable[..., list[llm.BatchResult]] = llm.run_batch,
) -> tuple[dict[str, bool], FictionReport]:
    """Return ({ol_id: is_fiction}, report) for every book.

    Subjects decide the clear cases; the model decides the rest (new ones only, the cache covers the
    rest). A book with no usable verdict is reported and treated as not fiction.
    """
    cache_path = cache_path or CACHE_PATH
    cache = _load_cache(cache_path)
    template = load_prompt("fiction_check")
    todo = books_needing_model(books, cache)

    if todo:
        requests = [
            llm.BatchRequest(
                f"f{b.ol_id}",
                [{"role": "user", "content": _prompt(template, b)}],
                max_tokens=MAX_TOKENS,
                thinking=llm.THINKING_OFF,
            )
            for b in todo
        ]
        digest = hashlib.sha1("|".join(sorted(r.custom_id for r in requests)).encode()).hexdigest()
        cost = estimate(
            [_prompt(template, b) for b in todo],
            output_tokens_each=EXPECTED_OUTPUT_TOKENS,
            batch=dispatch is llm.run_batch,
        ).usd
        results = dispatch(
            conn,
            purpose=PURPOSE,
            tag=f"{FICTION_PROMPT_VERSION}-{digest[:10]}",
            model=model,
            requests=requests,
            estimated_usd=cost,
            poll_seconds=poll_seconds,
            sleep=sleep,
        )
        by_id = {r.custom_id: r for r in results}
        for b in todo:
            result = by_id.get(f"f{b.ol_id}")
            if result is None or result.status != "succeeded":
                continue
            try:
                cache[b.ol_id] = parse_verdict(result.text or "").model_dump()
            except ValueError:
                logger.warning("fiction_check: unusable reply for %s", b.ol_id)
        _save_cache(cache_path, cache)

    report = FictionReport()
    verdicts: dict[str, bool] = {}
    for b in books:
        by_rules = classify_subjects(b.subjects)
        if by_rules is not None:
            verdicts[b.ol_id] = by_rules
            report.by_rules += 1
        elif b.ol_id in cache:
            verdicts[b.ol_id] = FictionVerdict(**cache[b.ol_id]).keep
            report.by_model += 1
        else:
            verdicts[b.ol_id] = False
            report.undecided.append(b.ol_id)
    logger.info(
        "fiction: %d by subjects, %d by the model, %d undecided (treated as not fiction)",
        report.by_rules,
        report.by_model,
        len(report.undecided),
    )
    return verdicts, report
