"""Open Library monthly dumps: a resumable downloader and streaming readers.

The Open Library API is for one-off lookups only, so the catalog comes from the dumps (CC0). They
are large (works ~4 GB compressed), tab-separated, sorted by key, and read as a stream, never loaded
whole. Line formats:
  works / authors : type <TAB> key <TAB> revision <TAB> last_modified <TAB> JSON
  reading-log     : work_key <TAB> edition_key <TAB> shelf <TAB> date
  ratings         : work_key <TAB> edition_key <TAB> rating <TAB> date
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from median.config import DUMPS_DIR, get_settings
from pipeline.sources import FetchedWork, clip_synopsis
from pipeline.sources.openlibrary import COVER_URL, MAX_SUBJECTS, clean_description

logger = logging.getLogger(__name__)

DUMP_NAMES = ("reading-log", "ratings", "authors", "works")  # smallest first
DUMP_URL = "https://openlibrary.org/data/ol_dump_{name}_latest.txt.gz"
CHUNK_BYTES = 1 << 20
LOG_EVERY_BYTES = 256 << 20
FREE_SPACE_MARGIN = 1 << 30  # keep 1 GB free beyond the download itself
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


# Download ----------------------------------------------------------------------------------------


def dump_path(name: str, dest_dir: Path | None = None) -> Path:
    """Return where the dump called `name` lives (or will be downloaded)."""
    return (dest_dir or DUMPS_DIR) / f"ol_dump_{name}_latest.txt.gz"


def _user_agent() -> str:
    contact = get_settings().contact_email
    if not contact:
        raise RuntimeError("CONTACT_EMAIL is not set; Open Library asks for it in the User-Agent.")
    return f"Median/0.1 (class project; contact: {contact})"


def _write_manifest(dest_dir: Path, name: str, size: int, last_modified: str | None) -> None:
    path = dest_dir / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest[name] = {
        "size": size,
        "last_modified": last_modified,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_dump(
    name: str, *, dest_dir: Path | None = None, http: httpx.Client | None = None
) -> Path:
    """Download one dump, resuming a partial file, and return its path.

    Skips the download when a complete file of the right size is already there. Checks free disk
    space first and verifies the final size. Raises OSError if the disk is too full.
    """
    dest_dir = dest_dir or DUMPS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dump_path(name, dest_dir)
    part = final.with_name(final.name + ".part")
    url = DUMP_URL.format(name=name)
    owns_client = http is None
    client = http or httpx.Client(
        headers={"User-Agent": _user_agent()}, timeout=60.0, follow_redirects=True
    )
    try:
        head = client.head(url)
        head.raise_for_status()
        total = int(head.headers["content-length"])
        last_modified = head.headers.get("last-modified")
        if final.exists() and final.stat().st_size == total:
            logger.info("%s already downloaded (%.2f GB)", final.name, total / 1e9)
            return final
        final.unlink(missing_ok=True)  # a different size means an older monthly dump
        have = part.stat().st_size if part.exists() else 0
        if have > total:
            part.unlink()
            have = 0
        needed = total - have
        free = shutil.disk_usage(dest_dir).free
        if free < needed + FREE_SPACE_MARGIN:
            raise OSError(
                f"Need {needed / 1e9:.1f} GB (+1 GB spare) for {name} but only {free / 1e9:.1f} GB "
                f"is free in {dest_dir}"
            )
        headers = {"Range": f"bytes={have}-"} if have else {}
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and have == total:  # already complete
                part.rename(final)
                return final
            response.raise_for_status()
            resumed = response.status_code == 206
            if have and not resumed:
                have = 0  # the server ignored Range, so start over
            written = have
            next_log = written + LOG_EVERY_BYTES
            with part.open("ab" if resumed else "wb") as out:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    out.write(chunk)
                    written += len(chunk)
                    if written >= next_log:
                        logger.info("  %s: %.1f / %.1f GB", name, written / 1e9, total / 1e9)
                        next_log += LOG_EVERY_BYTES
        if part.stat().st_size != total:
            raise OSError(
                f"{name}: got {part.stat().st_size} bytes, expected {total}. Rerun to resume."
            )
        part.rename(final)
        _write_manifest(dest_dir, name, total, last_modified)
        logger.info("Downloaded %s (%.2f GB)", final.name, total / 1e9)
        return final
    finally:
        if owns_client:
            client.close()


# Streaming readers -------------------------------------------------------------------------------


def _lines(path: Path) -> Iterator[bytes]:
    with gzip.open(path, "rb") as stream:
        yield from stream


def count_reading_log(path: Path) -> Counter[str]:
    """Return how many reading-log entries (all shelves) each work key has."""
    counts: Counter[bytes] = Counter()
    for line in _lines(path):
        tab = line.find(b"\t")
        if tab > 0:
            counts[line[:tab]] += 1
    return Counter({key.decode(): n for key, n in counts.items()})


def stream_records(path: Path, wanted: set[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (key, parsed JSON) for the wanted keys of a works/authors dump, then stop early.

    Only wanted lines are JSON-parsed, so a pass over the whole file stays cheap.
    """
    remaining = {key.encode() for key in wanted}
    for line in _lines(path):
        if not remaining:
            return
        first = line.find(b"\t")
        second = line.find(b"\t", first + 1)
        key = line[first + 1 : second]
        if key in remaining:
            remaining.discard(key)
            yield key.decode(), json.loads(line[line.rfind(b"\t") + 1 :])


def project_work(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce a works-dump record to the few fields the pipeline uses."""
    covers = [c for c in record.get("covers", []) if isinstance(c, int) and c > 0]
    return {
        "title": record.get("title", ""),
        "description": clean_description(record.get("description")),
        "subjects": [s for s in record.get("subjects", []) if isinstance(s, str)][:MAX_SUBJECTS],
        "author_keys": [
            a["author"]["key"]
            for a in record.get("authors", [])
            if isinstance(a, dict) and isinstance(a.get("author"), dict) and a["author"].get("key")
        ],
        "first_publish_date": record.get("first_publish_date"),
        "cover_id": covers[0] if covers else None,
    }


def project_author(record: dict[str, Any]) -> dict[str, Any]:
    return {"name": record.get("name") or record.get("personal_name") or ""}


class RecordCache:
    """An append-only JSONL cache of projected dump records, keyed by Open Library key.

    `get_many` streams the dump only for keys not seen before (including keys the dump lacks, which
    are cached as null), so the 4 GB works file is read once however often the step is rerun.
    """

    def __init__(self, path: Path, project: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._path = path
        self._project = project
        self._data: dict[str, dict[str, Any] | None] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                self._data[item["key"]] = item["record"]

    def get_many(self, keys: list[str], dump: Path) -> dict[str, dict[str, Any] | None]:
        missing = {k for k in keys if k not in self._data}
        if missing:
            logger.info("Reading %s for %d new key(s)...", dump.name, len(missing))
            found: dict[str, dict[str, Any] | None] = {
                key: self._project(record) for key, record in stream_records(dump, missing)
            }
            for key in missing:
                found.setdefault(key, None)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as out:
                for key, record in found.items():
                    out.write(json.dumps({"key": key, "record": record}) + "\n")
            self._data.update(found)
        return {k: self._data[k] for k in keys}


# Normalising -------------------------------------------------------------------------------------


def parse_year(value: Any) -> int | None:
    """Return the first plausible 4-digit year in an OL date string, else None."""
    match = YEAR.search(value) if isinstance(value, str) else None
    return int(match.group(1)) if match else None


def has_description(projected: dict[str, Any] | None, min_chars: int) -> bool:
    """True when the projected work has a description of at least `min_chars` characters."""
    return bool(projected) and len(projected["description"]) >= min_chars


def to_fetched_work(
    key: str,
    projected: dict[str, Any],
    author_names: dict[str, str],
    popularity: float | None,
) -> FetchedWork:
    """Build a FetchedWork from a projected works record plus resolved author names."""
    cover_id = projected["cover_id"]
    return FetchedWork(
        source="openlibrary",
        external_id=key.rsplit("/", 1)[-1],
        media_type="book",
        title=projected["title"],
        year=parse_year(projected["first_publish_date"]),
        creators=[author_names[k] for k in projected["author_keys"] if author_names.get(k)],
        synopsis=clip_synopsis(projected["description"]),
        subjects=projected["subjects"],
        cover_url=COVER_URL.format(cover_id=cover_id) if cover_id else None,
        popularity_raw=popularity,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: download the dumps (all by default). Resumable; safe to rerun."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Open Library dumps into data/dumps/.")
    parser.add_argument("names", nargs="*", choices=DUMP_NAMES, help="default: all four")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for name in args.names or DUMP_NAMES:
        download_dump(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
