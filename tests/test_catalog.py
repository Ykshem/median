from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from pipeline import catalog
from pipeline.sources import ol_dumps, tmdb
from tests.fixtures.source_payloads import tmdb_details


def _tmdb_client(tmp_path: Path, payloads: dict[int, dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        movie_id = int(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, json=payloads[movie_id])

    return tmdb.make_client(
        transport=httpx.MockTransport(handler), cache_dir=tmp_path / "c", sleep=lambda s: None
    )


def test_load_tmdb_movies_upserts_and_skips_empty_overviews(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    payloads = {
        1: {**tmdb_details(1, "One", "2001-01-01"), "popularity": 55.5},
        2: tmdb_details(2, "Two", "2002-02-02", overview=""),
        3: tmdb_details(3, "Three", "2003-03-03"),
    }
    with _tmdb_client(tmp_path, payloads) as client:
        report = catalog.load_tmdb_movies(db, client, [1, 2, 3])
    assert report.loaded == 2 and len(report.skipped) == 1 and "tmdb:2" in report.skipped[0]
    rows = db.execute("SELECT title, popularity_raw FROM works ORDER BY title").fetchall()
    assert [(r["title"], r["popularity_raw"]) for r in rows] == [("One", 55.5), ("Three", None)]
    assert db.execute("SELECT COUNT(*) FROM external_ids WHERE source = 'tmdb'").fetchone()[0] == 2


# Open Library -----------------------------------------------------------------------------------


def _write_dump(path: Path, lines: list[str]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _line(kind: str, key: str, record: dict) -> str:
    return "\t".join([f"/type/{kind}", key, "1", "2024-01-01", json.dumps(record)])


@pytest.fixture
def dump_files(tmp_path: Path) -> dict[str, Path]:
    long_text = "A long enough description of the story. " * 3
    works = _write_dump(
        tmp_path / "works.gz",
        [
            _line(
                "work",
                "/works/OL1W",
                {
                    "title": "Described",
                    "description": long_text,
                    "authors": [{"author": {"key": "/authors/OL7A"}}],
                    "first_publish_date": "1977",
                    "subjects": ["Horror"],
                },
            ),
            _line("work", "/works/OL2W", {"title": "No description", "subjects": ["x"]}),
            _line("work", "/works/OL3W", {"title": "Short", "description": "Too short."}),
        ],
    )
    authors = _write_dump(
        tmp_path / "authors.gz", [_line("author", "/authors/OL7A", {"name": "A. Author"})]
    )
    return {"works": works, "authors": authors}


def _load(db: sqlite3.Connection, tmp_path: Path, dump_files: dict[str, Path], ids: list[str]):
    return catalog.load_openlibrary_books(
        db,
        ids,
        reading_counts={"/works/OL1W": 321, "/works/OL2W": 9},
        works_cache=ol_dumps.RecordCache(tmp_path / "w.jsonl", ol_dumps.project_work),
        authors_cache=ol_dumps.RecordCache(tmp_path / "a.jsonl", ol_dumps.project_author),
        works_dump=dump_files["works"],
        authors_dump=dump_files["authors"],
        min_description_chars=40,
    )


def test_load_openlibrary_books_requires_a_description(
    db: sqlite3.Connection, tmp_path: Path, dump_files: dict[str, Path]
) -> None:
    report = _load(db, tmp_path, dump_files, ["OL1W", "OL2W", "OL3W", "OL404W"])
    assert report.loaded == 1 and len(report.skipped) == 3
    row = db.execute("SELECT * FROM works").fetchone()
    assert (row["title"], row["year"], row["popularity_raw"]) == ("Described", 1977, 321.0)
    assert row["media_type"] == "book" and json.loads(row["creators"]) == ["A. Author"]
    assert db.execute("SELECT external_id FROM external_ids").fetchone()[0] == "OL1W"


def test_load_openlibrary_books_does_not_need_the_dumps_on_a_rerun(
    db: sqlite3.Connection, tmp_path: Path, dump_files: dict[str, Path]
) -> None:
    _load(db, tmp_path, dump_files, ["OL1W"])
    for dump in dump_files.values():
        dump.unlink()
    report = _load(db, tmp_path, dump_files, ["OL1W"])  # served from the record caches
    assert report.loaded == 1
    assert db.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 1


# Wikidata links ---------------------------------------------------------------------------------


def _two_works(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        INSERT INTO works (id, media_type, title) VALUES (1, 'movie', 'Film'), (2, 'book', 'Book'),
                                                         (3, 'movie', 'Other film');
        INSERT INTO external_ids VALUES (1, 'tmdb', '694'), (2, 'openlibrary', 'OL1W'),
                                        (3, 'tmdb', '5');
        """
    )


def test_wikidata_links_need_both_ends_in_the_catalog(db: sqlite3.Connection) -> None:
    _two_works(db)
    counts = catalog.link_wikidata(
        db,
        adaptations=[(694, "OL1W"), (694, "OL999W"), (12345, "OL1W")],
        series=[(5, "OL1W", "Q42")],
    )
    assert counts == {"adaptation_of": 1, "same_franchise": 1}
    rows = db.execute("SELECT * FROM work_links ORDER BY link_type").fetchall()
    assert [(r["from_work"], r["to_work"], r["link_type"]) for r in rows] == [
        (1, 2, "adaptation_of"),
        (3, 2, "same_franchise"),
    ]
    assert all(
        r["source"] == "wikidata" and r["verified"] == 1 and r["confidence"] == 1.0 for r in rows
    )


def test_wikidata_replaces_an_unverified_candidate_and_is_idempotent(
    db: sqlite3.Connection,
) -> None:
    _two_works(db)
    db.execute("INSERT INTO work_links VALUES (1, 2, 'adaptation_of', 'title_match', 0.5, 0)")
    for _ in range(2):
        catalog.link_wikidata(db, [(694, "OL1W")], [])
    (row,) = db.execute("SELECT * FROM work_links").fetchall()
    assert (row["source"], row["confidence"], row["verified"]) == ("wikidata", 1.0, 1)
