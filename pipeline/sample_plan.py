"""Choose which ~1,000 movies and ~1,000 fiction books go in the --sample catalog.

The Wikidata adaptation pairs decide which works to fetch, so selection runs before the fetch steps.
It is a pure function over its inputs (reading counts, pairs, a "has description?" lookup and a TMDB
popularity list are passed in), so it can be tested offline and gives the same plan every time.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleSpec:
    movies: int = 1000
    books: int = 1000
    min_pairs: int = 100  # adaptation pairs that must survive; more are selected as a buffer
    pair_buffer: float = 1.6  # some pairs lose an end (no overview / no description)
    max_films_per_book: int = 2  # stops one folk tale with 30 films flooding the pairs
    min_description_chars: int = 80
    book_candidates: int = 6000  # top books by reading-log count considered for the fill


@dataclass(frozen=True)
class SamplePlan:
    movies: list[int]  # TMDB movie ids
    books: list[str]  # Open Library work ids, e.g. "OL82563W"
    pairs: list[tuple[int, str]]  # (tmdb id, OL work id) adaptation pairs, forced + Wikidata
    spec: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> SamplePlan:
        raw = json.loads(text)
        return cls(raw["movies"], raw["books"], [tuple(p) for p in raw["pairs"]], raw["spec"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SamplePlan:
        return cls.from_json(path.read_text(encoding="utf-8"))


def bare_id(key: str) -> str:
    """Return the Open Library id from a key: '/works/OL1W' -> 'OL1W' (bare ids pass through)."""
    return key.rsplit("/", 1)[-1]


def select_sample(
    spec: SampleSpec,
    *,
    forced_movies: list[int],
    forced_books: list[str],
    forced_pairs: list[tuple[int, str]],
    wikidata_pairs: list[tuple[int, str]],
    reading_counts: Mapping[str, int],
    has_description: Callable[[list[str]], Mapping[str, bool]],
    is_fiction: Callable[[list[str]], Mapping[str, bool]],
    discover_movies: Callable[[int], list[int]],
) -> SamplePlan:
    """Return the sample plan. Books are fiction only (ADR 0002); movies are unrestricted.

    forced_*: always included (the review set and the gold pairs), unless the book is not fiction,
    which is logged and drops its pairs too. reading_counts: OL work key -> reading-log entries.
    has_description: OL ids -> whether the work has a usable description. is_fiction: OL ids ->
    whether the book is fiction; it is asked only about books that could be selected, because the
    answer can cost an LLM call. discover_movies(n): the n most popular TMDB movie ids.
    """
    counts = {bare_id(k): n for k, n in reading_counts.items()}
    by_book: dict[str, list[int]] = {}
    for tmdb, book in sorted(wikidata_pairs):
        by_book.setdefault(book, []).append(tmdb)
    popular_pair_books = sorted(by_book, key=lambda b: (-counts.get(b, 0), b))

    target_pairs = math.ceil(spec.min_pairs * spec.pair_buffer)
    pair_candidates = popular_pair_books[: max(target_pairs * 4, 200)]  # headroom for unusable ends
    top_books = [
        b for b, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: spec.book_candidates]
    ]
    usable = has_description(list(dict.fromkeys([*forced_books, *pair_candidates, *top_books])))
    forced = list(dict.fromkeys([*forced_books, *(book for _, book in forced_pairs)]))
    selectable = [b for b in dict.fromkeys([*pair_candidates, *top_books]) if usable.get(b)]
    fiction = is_fiction(list(dict.fromkeys([*forced, *selectable])))
    dropped = [b for b in forced if not fiction.get(b)]
    if dropped:
        logger.warning("Forced books dropped because they are not fiction: %s", ", ".join(dropped))
    forced_books = [b for b in forced_books if fiction.get(b)]
    forced_pairs = [(t, b) for t, b in forced_pairs if fiction.get(b)]
    usable = {b: bool(usable.get(b)) and bool(fiction.get(b)) for b in usable}

    pairs: dict[tuple[int, str], None] = dict.fromkeys(forced_pairs)
    for book in pair_candidates:
        if len(pairs) >= target_pairs:
            break
        if not usable.get(book):
            continue
        for tmdb in by_book[book][: spec.max_films_per_book]:
            if len(pairs) < target_pairs:
                pairs.setdefault((tmdb, book))

    books: dict[str, None] = dict.fromkeys(forced_books)
    books.update(dict.fromkeys(book for _, book in pairs))
    for book in top_books:
        if len(books) >= spec.books:
            break
        if usable.get(book):
            books.setdefault(book)

    movies: dict[int, None] = dict.fromkeys(forced_movies)
    movies.update(dict.fromkeys(tmdb for tmdb, _ in pairs))
    if len(movies) < spec.movies:
        for movie_id in discover_movies(spec.movies + len(movies)):
            if len(movies) >= spec.movies:
                break
            movies.setdefault(movie_id)

    return SamplePlan(
        movies=list(movies),
        books=list(books),
        pairs=list(pairs),
        spec={k: v for k, v in asdict(spec).items()},
    )
