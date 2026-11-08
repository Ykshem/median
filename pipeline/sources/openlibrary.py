"""Open Library (books): one-off API lookups for the small review set.

The API is not for bulk download: the 20k-book catalog will stream the monthly dumps instead (a
later M1 step). This module only does a handful of single lookups, with a User-Agent that names the
app and the project's CONTACT_EMAIL setting, at well under 3 requests/s.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from median.config import get_settings
from pipeline.sources import FetchedWork, SkipWork, clip_synopsis, normalize_title
from pipeline.sources.http import CachedClient

logger = logging.getLogger(__name__)

BASE_URL = "https://openlibrary.org"
COVER_URL = "https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
MAX_SUBJECTS = 12


def make_client(**kwargs: Any) -> CachedClient:
    """Return an Open Library client. Needs CONTACT_EMAIL (OL asks for it in the User-Agent)."""
    contact = get_settings().contact_email
    if not contact:
        raise RuntimeError(
            "CONTACT_EMAIL is not set. Open Library asks for a contact in the User-Agent; "
            "set it in .env."
        )
    return CachedClient(
        "openlibrary",
        BASE_URL,
        headers={"User-Agent": f"Median/0.1 (class project; contact: {contact})"},
        min_interval=0.4,
        **kwargs,
    )


def clean_description(value: Any) -> str:
    """Return plain description text from OL's str-or-{type, value} field, minus OL boilerplate."""
    if isinstance(value, dict):
        value = value.get("value", "")
    if not isinstance(value, str):
        return ""
    text = value.replace("\r\n", "\n")
    text = re.split(r"\n-{5,}", text)[0]  # OL appends "----------" blocks (sources, contents)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links -> their text
    return text.strip()


def _pick_doc(docs: list[dict[str, Any]], title: str, author: str) -> tuple[dict[str, Any], bool]:
    """Return (best search doc, is_exact); exact = same normalised title and the author surname."""
    surname = normalize_title(author).split()[-1]
    wanted = normalize_title(title)
    for doc in docs:
        names = {t for a in doc.get("author_name", []) for t in normalize_title(a).split()}
        if normalize_title(doc.get("title", "")) == wanted and surname in names:
            return doc, True
    return docs[0], False


def _credited_authors(doc: dict[str, Any], author: str) -> list[str]:
    """Return the doc's authors that share the requested author's surname.

    OL's search `author_name` also lists people from edition records (translators, illustrators,
    study guides). Falls back to every listed name if none match.
    """
    surname = normalize_title(author).split()[-1]
    names = list(doc.get("author_name", []))
    matching = [n for n in names if surname in normalize_title(n).split()]
    return matching or names


def find_book(client: CachedClient, title: str, author: str) -> FetchedWork:
    """Return the book matching `title` and `author`, using the search doc plus the work record.

    Raises SkipWork if nothing is found or the work has no description to describe it from. A
    non-exact pick is returned with `match_note` set.
    """
    docs = client.get_json(
        "/search.json",
        {
            "title": title,
            "author": author,
            "fields": "key,title,author_name,first_publish_year,cover_i",
            "limit": 5,
        },
    ).get("docs", [])
    if not docs:
        raise SkipWork(f"Open Library search found nothing for {title!r} by {author}")
    doc, exact = _pick_doc(docs, title, author)
    work_id = doc["key"].rsplit("/", 1)[-1]
    raw = client.get_json(f"/works/{work_id}.json")
    description = clean_description(raw.get("description"))
    if not description:
        raise SkipWork(f"Open Library {work_id} ({doc.get('title')}) has no description")
    cover_id = doc.get("cover_i")
    work = FetchedWork(
        source="openlibrary",
        external_id=work_id,
        media_type="book",
        title=doc.get("title") or raw.get("title", title),
        year=doc.get("first_publish_year"),
        creators=_credited_authors(doc, author),
        synopsis=clip_synopsis(description),
        subjects=list(raw.get("subjects", []))[:MAX_SUBJECTS],
        cover_url=COVER_URL.format(cover_id=cover_id) if cover_id else None,
    )
    if exact:
        return work
    note = f"weak match for {title!r} by {author}: picked {work.title!r} by {work.creators}"
    logger.warning(note)
    return dataclasses.replace(work, match_note=note)
