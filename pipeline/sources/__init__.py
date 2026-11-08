"""Data-source fetchers. Each returns `FetchedWork`, so later steps are source-agnostic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_SYNOPSIS_CHARS = 1200  # keeps prompts (and cost) bounded; long blurbs add little signal


class SkipWork(Exception):
    """Raised when a work can't be used (not found, or no synopsis to describe it from)."""


@dataclass(frozen=True)
class FetchedWork:
    """A catalog item normalised from TMDB or Open Library."""

    source: str  # 'tmdb' | 'openlibrary' (matches external_ids.source)
    external_id: str
    media_type: str  # 'movie' | 'book'
    title: str
    year: int | None
    creators: list[str]
    synopsis: str
    subjects: list[str]
    cover_url: str | None = None
    popularity_raw: float | None = None  # TMDB popularity / OL reading-log count
    match_note: str | None = field(default=None, compare=False)  # set when the match is weak


def normalize_title(text: str) -> str:
    """Return a comparison key: casefolded, '&' as 'and', punctuation dropped, spaces collapsed."""
    text = text.casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())


def clip_synopsis(text: str) -> str:
    """Return `text` trimmed to MAX_SYNOPSIS_CHARS, with an ellipsis if it was cut."""
    text = " ".join(text.split())
    return text if len(text) <= MAX_SYNOPSIS_CHARS else text[:MAX_SYNOPSIS_CHARS].rstrip() + "…"
