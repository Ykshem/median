from __future__ import annotations

import sqlite3

import pytest

from pipeline import sample_seed
from pipeline.sources import FetchedWork, SkipWork


def _movie(seed: sample_seed.SeedMovie) -> FetchedWork:
    return FetchedWork(
        "tmdb",
        f"m:{seed.title}",
        "movie",
        seed.title,
        seed.year,
        ["Dir"],
        f"About {seed.title}",
        [],
    )


def _book(seed: sample_seed.SeedBook) -> FetchedWork:
    return FetchedWork(
        "openlibrary", f"b:{seed.title}", "book", seed.title, 1990, [seed.author], "About it", []
    )


@pytest.fixture
def stub_sources(monkeypatch: pytest.MonkeyPatch):
    def install(*, missing_movies: set[str] = frozenset(), missing_books: set[str] = frozenset()):
        def find_movie(client, title, year):
            if title in missing_movies:
                raise SkipWork(f"{title} unavailable")
            return _movie(sample_seed.SeedMovie(title, year))

        def find_book(client, title, author):
            if title in missing_books:
                raise SkipWork(f"{title} has no description")
            return _book(sample_seed.SeedBook(title, author))

        monkeypatch.setattr(sample_seed, "find_movie", find_movie)
        monkeypatch.setattr(sample_seed, "find_book", find_book)

    return install


def test_seed_lists_have_the_agreed_shape() -> None:
    assert len(sample_seed.PAIRS) == 5
    assert len(sample_seed.MOVIES) == len(sample_seed.BOOKS) == 20
    titles = [m.title for m in sample_seed.MOVIES] + [p.movie.title for p in sample_seed.PAIRS]
    assert len(titles) == len(set(titles)), "duplicate movie in the seed"
    book_titles = [b.title for b in sample_seed.BOOKS] + [p.book.title for p in sample_seed.PAIRS]
    assert len(book_titles) == len(set(book_titles)), "duplicate book in the seed"


def test_load_sample_builds_25_movies_25_books_and_5_pairs(
    db: sqlite3.Connection, stub_sources
) -> None:
    stub_sources()
    result = sample_seed.load_sample(db, object(), object())
    assert result.failures == []
    assert len(result.entries) == 50
    assert db.execute("SELECT COUNT(*) FROM works WHERE media_type = 'movie'").fetchone()[0] == 25
    assert db.execute("SELECT COUNT(*) FROM works WHERE media_type = 'book'").fetchone()[0] == 25
    pairs = {e.pair for e in result.entries if e.pair}
    assert len(pairs) == 5
    for name in pairs:
        kinds = sorted(e.fetched.media_type for e in result.entries if e.pair == name)
        assert kinds == ["book", "movie"]


def test_load_sample_is_idempotent(db: sqlite3.Connection, stub_sources) -> None:
    stub_sources()
    sample_seed.load_sample(db, object(), object())
    sample_seed.load_sample(db, object(), object())
    assert db.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 50
    assert db.execute("SELECT COUNT(*) FROM external_ids").fetchone()[0] == 50


def test_unusable_standalone_works_are_replaced_from_the_reserves(
    db: sqlite3.Connection, stub_sources
) -> None:
    stub_sources(missing_movies={"Parasite"}, missing_books={"Piranesi", "Circe"})
    result = sample_seed.load_sample(db, object(), object())
    assert len(result.failures) == 3
    assert db.execute("SELECT COUNT(*) FROM works WHERE media_type = 'movie'").fetchone()[0] == 25
    assert db.execute("SELECT COUNT(*) FROM works WHERE media_type = 'book'").fetchone()[0] == 25
    assert db.execute("SELECT 1 FROM works WHERE title = 'Inception'").fetchone()
    assert db.execute("SELECT 1 FROM works WHERE title = 'The Midnight Library'").fetchone()
    assert db.execute("SELECT 1 FROM works WHERE title = 'Babel'").fetchone()


def test_a_missing_pair_member_is_reported_not_replaced(
    db: sqlite3.Connection, stub_sources
) -> None:
    stub_sources(missing_books={"The Martian"})
    result = sample_seed.load_sample(db, object(), object())
    assert any("The Martian" in f for f in result.failures)
    assert len(result.entries) == 49
    assert db.execute("SELECT COUNT(*) FROM works WHERE media_type = 'book'").fetchone()[0] == 24


def test_works_are_searchable_after_loading(db: sqlite3.Connection, stub_sources) -> None:
    stub_sources()
    sample_seed.load_sample(db, object(), object())
    hit = db.execute("SELECT rowid FROM works_fts WHERE works_fts MATCH 'parasite'").fetchall()
    assert len(hit) == 1
