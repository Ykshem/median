"""Checks on the gold evaluation files.

The format checks are plain unit tests. The last test is marked `eval` and needs the built DB.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from median.config import GOLD_ADAPTATIONS_PATH, GOLD_VIBES_PATH, get_settings
from median.db import connect
from pipeline import gold
from pipeline.sources import FetchedWork, SkipWork
from pipeline.sources import openlibrary as ol_source
from pipeline.sources import tmdb as tmdb_source


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# The checked-in files --------------------------------------------------------------------------


def test_gold_adaptations_file_is_well_formed() -> None:
    rows = _rows(GOLD_ADAPTATIONS_PATH)
    assert len(rows) >= 50
    assert list(rows[0]) == gold.ADAPTATION_FIELDS
    assert all(re.fullmatch(r"OL\d+W", r["book_ol_id"]) for r in rows)
    assert all(r["movie_tmdb_id"].isdigit() for r in rows)
    assert all(r["in_wikidata"] in {"True", "False"} for r in rows)
    pairs = [(r["book_ol_id"], r["movie_tmdb_id"]) for r in rows]
    assert len(pairs) == len(set(pairs)), "duplicate pair"
    # both groups are present, so the eval can report Wikidata-covered vs not
    assert {r["in_wikidata"] for r in rows} == {"True", "False"}


def test_gold_vibes_file_is_well_formed_and_marked_as_proposed() -> None:
    rows = _rows(GOLD_VIBES_PATH)
    assert len(rows) >= 30
    assert list(rows[0]) == gold.VIBE_FIELDS
    ref = re.compile(r"(tmdb:\d+|ol:OL\d+W)")
    for r in rows:
        assert ref.fullmatch(r["seed_ref"]) and ref.fullmatch(r["target_ref"])
        assert r["seed_ref"].split(":")[0] != r["target_ref"].split(":")[0], "must cross media"
        assert r["why"].strip()
        assert r["status"] == "proposed", "flip to 'agreed' only after the team signs off"
    refs = [(r["seed_ref"], r["target_ref"]) for r in rows]
    assert len(refs) == len(set(refs))


def test_no_vibe_pair_is_really_an_adaptation_pair() -> None:
    adaptations = {
        (f"ol:{r['book_ol_id']}", f"tmdb:{r['movie_tmdb_id']}")
        for r in _rows(GOLD_ADAPTATIONS_PATH)
    }
    for r in _rows(GOLD_VIBES_PATH):
        assert (r["seed_ref"], r["target_ref"]) not in adaptations
        assert (r["target_ref"], r["seed_ref"]) not in adaptations


def test_read_gold_adaptations_returns_typed_pairs() -> None:
    pairs = gold.read_gold_adaptations()
    assert len(pairs) >= 50
    assert all(isinstance(p.movie_tmdb_id, int) and p.book_ol_id.startswith("OL") for p in pairs)
    assert gold.read_gold_adaptations(Path("does-not-exist.csv")) == []


@pytest.mark.eval
def test_every_gold_id_exists_in_the_built_catalog() -> None:
    """Needs `pipeline.build --sample` first: the sample is built to contain all gold ids."""
    conn = connect(get_settings().db_path)
    try:
        have = {(r["source"], r["external_id"]) for r in conn.execute("SELECT * FROM external_ids")}
    finally:
        conn.close()
    if not have:
        pytest.skip("no catalog built yet")
    missing = []
    for r in _rows(GOLD_ADAPTATIONS_PATH):
        for ref in (("openlibrary", r["book_ol_id"]), ("tmdb", r["movie_tmdb_id"])):
            if ref not in have:
                missing.append(ref)
    for r in _rows(GOLD_VIBES_PATH):
        for field in ("seed_ref", "target_ref"):
            kind, ident = r[field].split(":")
            if ("openlibrary" if kind == "ol" else "tmdb", ident) not in have:
                missing.append((kind, ident))
    assert not missing, f"gold ids missing from the catalog: {missing[:10]}"


# Drafting ---------------------------------------------------------------------------------------


def _movie(title: str, ext: str = "1", note: str | None = None) -> FetchedWork:
    return FetchedWork("tmdb", ext, "movie", title, 2000, ["D"], "s", [], match_note=note)


def _book(title: str, ext: str = "OL1W", note: str | None = None) -> FetchedWork:
    return FetchedWork(
        "openlibrary", ext, "book", title, 1990, ["Au Thor"], "s", [], match_note=note
    )


def test_draft_adaptations_accepts_only_exact_unique_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def find_movie(client, title, year):
        if title == "Missing":
            raise SkipWork("nothing found")
        return _movie(
            title,
            ext={"Good": "1", "Weak": "2", "Dup": "1"}[title],
            note="weak" if title == "Weak" else None,
        )

    monkeypatch.setattr(tmdb_source, "find_movie", find_movie)
    monkeypatch.setattr(ol_source, "find_book", lambda client, title, author: _book(title, "OL1W"))
    seeds = [gold.A(t, 2000, "B", "Au Thor") for t in ("Good", "Weak", "Missing", "Dup")]
    rows, rejected = gold.draft_adaptations(object(), object(), {(1, "OL1W")}, seeds)
    assert [r["movie_title"] for r in rows] == ["Good"]
    assert rows[0]["in_wikidata"] is True and rows[0]["book_ol_id"] == "OL1W"
    assert len(rejected) == 3
    assert any("not an exact match" in r for r in rejected) and any(
        "duplicate" in r for r in rejected
    )


def test_draft_vibes_resolves_both_sides_and_marks_them_proposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tmdb_source, "find_movie", lambda c, t, y: _movie(t, "694"))
    monkeypatch.setattr(ol_source, "find_book", lambda c, t, a: _book(t, "OL5W"))
    seed = gold.V(("movie", "Hereditary", 2018), ("book", "Hill House", "S. Jackson"), "dread")
    rows, rejected = gold.draft_vibes(object(), object(), [seed])
    assert rejected == []
    assert rows == [
        {
            "seed_ref": "tmdb:694",
            "target_ref": "ol:OL5W",
            "seed_title": "Hereditary",
            "target_title": "Hill House",
            "why": "dread",
            "status": "proposed",
        }
    ]


def test_write_csv_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "x.csv"
    gold.write_csv(path, ["a", "b"], [{"a": "has, comma", "b": 2}])
    assert _rows(path) == [{"a": "has, comma", "b": "2"}]


def test_read_gold_vibe_refs_splits_movies_and_books(tmp_path: Path) -> None:
    path = tmp_path / "v.csv"
    path.write_text(
        "seed_ref,target_ref,seed_title,target_title,why,status\n"
        "tmdb:5,ol:OL1W,a,b,x,proposed\n"
        "ol:OL2W,tmdb:5,c,a,y,proposed\n",
        encoding="utf-8",
    )
    assert gold.read_gold_vibe_refs(path) == ([5], ["OL1W", "OL2W"])
    assert gold.read_gold_vibe_refs(tmp_path / "missing.csv") == ([], [])
