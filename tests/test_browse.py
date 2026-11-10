from __future__ import annotations

import sqlite3

import pytest

from median import browse
from median.browse import SearchFilters, fts_query

SHINING_FILM, SHINING_BOOK = 1, 11
HILL_HOUSE_BOOK = 12
HOBBIT_BOOK = 16
REMAINS_FILM, REMAINS_BOOK = 8, 18  # link between them is unverified


def ids(works) -> list[int]:
    return [w.id for w in works]


# fts_query ------------------------------------------------------------------------------------


def test_fts_query_quotes_every_word_and_prefixes_the_last() -> None:
    assert fts_query("dark tower") == '"dark" "tower"*'


def test_fts_query_neutralizes_fts_syntax() -> None:
    assert fts_query('NEAR(a b) OR "x" -y*') == '"NEAR" "a" "b" "OR" "x" "y"*'


@pytest.mark.parametrize("text", ["", "   ", "***", '"" ()'])
def test_fts_query_is_none_without_words(text: str) -> None:
    assert fts_query(text) is None


# search ---------------------------------------------------------------------------------------


def test_empty_search_lists_everything_most_popular_first(mini_catalog, db: sqlite3.Connection):
    works = browse.search_works(db, SearchFilters(), limit=100)
    assert len(works) == 20
    pops = [w.popularity for w in works]
    assert pops == sorted(pops, reverse=True)
    assert browse.count_matches(db, SearchFilters()) == 20


def test_title_search_finds_both_media(mini_catalog, db: sqlite3.Connection):
    found = browse.search_works(db, SearchFilters(query="shining"))
    assert set(ids(found)) == {SHINING_FILM, SHINING_BOOK}


def test_prefix_search_matches_a_partial_last_word(mini_catalog, db: sqlite3.Connection):
    assert HILL_HOUSE_BOOK in ids(browse.search_works(db, SearchFilters(query="haunting of hil")))


def test_search_matches_creators(mini_catalog, db: sqlite3.Connection):
    work = browse.get_work(db, SHINING_BOOK)
    assert work is not None and work.creators
    surname = work.creators[0].split()[-1]
    assert SHINING_BOOK in ids(browse.search_works(db, SearchFilters(query=surname)))


def test_hostile_query_does_not_raise(mini_catalog, db: sqlite3.Connection):
    for text in ['"', "a OR", "title:", "(", "NEAR(", "'; DROP TABLE works; --"]:
        browse.search_works(db, SearchFilters(query=text))
    assert browse.count_matches(db, SearchFilters()) == 20


def test_media_filter(mini_catalog, db: sqlite3.Connection):
    books = browse.search_works(db, SearchFilters(media_type="book"), limit=100)
    assert len(books) == 10 and {w.media_type for w in books} == {"book"}


def test_unknown_media_type_or_sort_is_rejected(db: sqlite3.Connection):
    with pytest.raises(ValueError):
        browse.search_works(db, SearchFilters(media_type="tv"))
    with pytest.raises(ValueError):
        browse.search_works(db, SearchFilters(sort="random"))


def test_year_range_is_inclusive(mini_catalog, db: sqlite3.Connection):
    works = browse.search_works(db, SearchFilters(year_min=1977, year_max=1980), limit=100)
    assert {w.year for w in works} <= {1977, 1978, 1979, 1980}
    assert {SHINING_FILM, SHINING_BOOK} <= set(ids(works))


def test_mood_filter_is_case_insensitive(mini_catalog, db: sqlite3.Connection):
    works = browse.search_works(db, SearchFilters(mood="EERIE"), limit=100)
    assert set(ids(works)) == {1, 2, 12}


def test_filters_combine(mini_catalog, db: sqlite3.Connection):
    works = browse.search_works(db, SearchFilters(query="shining", media_type="book"))
    assert ids(works) == [SHINING_BOOK]


def test_sorts(mini_catalog, db: sqlite3.Connection):
    newest = browse.search_works(db, SearchFilters(sort="year_newest"), limit=100)
    assert [w.year for w in newest] == sorted((w.year for w in newest), reverse=True)
    oldest = browse.search_works(db, SearchFilters(sort="year_oldest"), limit=100)
    assert [w.year for w in oldest] == sorted(w.year for w in oldest)
    titles = [
        w.title.lower() for w in browse.search_works(db, SearchFilters(sort="title"), limit=100)
    ]
    assert titles == sorted(titles)


def test_year_sort_puts_unknown_years_last(mini_catalog, db: sqlite3.Connection):
    db.execute("UPDATE works SET year = NULL WHERE id = ?", (SHINING_BOOK,))
    for sort in ("year_newest", "year_oldest"):
        works = browse.search_works(db, SearchFilters(sort=sort), limit=100)
        assert works[-1].id == SHINING_BOOK


def test_paging_is_stable_and_complete(mini_catalog, db: sqlite3.Connection):
    f = SearchFilters(sort="title")
    pages = [browse.search_works(db, f, limit=7, offset=o) for o in (0, 7, 14)]
    flat = [w.id for p in pages for w in p]
    assert len(flat) == 20 and len(set(flat)) == 20


def test_bad_paging_is_rejected(db: sqlite3.Connection):
    with pytest.raises(ValueError):
        browse.search_works(db, SearchFilters(), limit=0)
    with pytest.raises(ValueError):
        browse.search_works(db, SearchFilters(), offset=-1)


def test_list_moods_and_year_bounds(mini_catalog, db: sqlite3.Connection):
    moods = browse.list_moods(db, limit=5)
    assert len(moods) == 5 and all(m == m.lower() for m in moods)
    assert browse.year_bounds(db) == (1934, 2020)


def test_year_bounds_of_an_empty_catalog(db: sqlite3.Connection):
    assert browse.year_bounds(db) is None


# work detail ----------------------------------------------------------------------------------


def test_get_work_parses_json_columns(mini_catalog, db: sqlite3.Connection):
    work = browse.get_work(db, SHINING_FILM)
    assert work is not None
    assert work.title == "The Shining" and work.media_type == "movie"
    assert isinstance(work.creators, list) and isinstance(work.subjects, list)
    assert browse.get_work(db, 9999) is None


def test_get_card_validates(mini_catalog, db: sqlite3.Connection):
    card = browse.get_card(db, SHINING_FILM)
    assert card is not None and "eerie" in card.mood
    assert browse.get_card(db, 9999) is None
    db.execute("UPDATE vibe_cards SET card_json = '{bad' WHERE work_id = ?", (SHINING_FILM,))
    assert browse.get_card(db, SHINING_FILM) is None


def test_links_are_direction_aware(mini_catalog, db: sqlite3.Connection):
    film = browse.linked_works(db, SHINING_FILM)
    assert [(lw.work.id, lw.relation) for lw in film] == [(SHINING_BOOK, "Adapted from")]
    book = browse.linked_works(db, SHINING_BOOK)
    assert [(lw.work.id, lw.relation) for lw in book] == [(SHINING_FILM, "Adapted as")]


def test_unverified_links_are_hidden(mini_catalog, db: sqlite3.Connection):
    assert browse.linked_works(db, REMAINS_FILM) == []
    assert browse.linked_works(db, REMAINS_BOOK) == []


def test_links_are_ordered_adaptation_first(mini_catalog, db: sqlite3.Connection):
    # Fellowship book (15): adaptation of film 5, same franchise as The Hobbit (16)
    types = [lw.link_type for lw in browse.linked_works(db, 15)]
    assert types == ["adaptation_of", "same_franchise"]
    labels = {lw.work.id: lw.relation for lw in browse.linked_works(db, 15)}
    assert labels[HOBBIT_BOOK] == "Same series"
    assert {lw.relation for lw in browse.linked_works(db, 19)} == {
        "Adapted as",
        "Same creator",
    }


def test_similar_vibe_is_the_other_medium_and_excludes_links(mini_catalog, db: sqlite3.Connection):
    work = browse.get_work(db, SHINING_FILM)
    assert work is not None
    similar = browse.similar_vibe(db, work, k=6)
    assert similar, "a horror film should have book neighbours"
    assert {s.work.media_type for s in similar} == {"book"}
    assert SHINING_BOOK not in ids(s.work for s in similar)  # linked: has its own section
    assert similar[0].work.id == HILL_HOUSE_BOOK  # the gothic-horror cluster
    assert all(s.recommendation.reasons for s in similar)


def test_similar_vibe_respects_k(mini_catalog, db: sqlite3.Connection):
    work = browse.get_work(db, SHINING_BOOK)
    assert work is not None
    assert len(browse.similar_vibe(db, work, k=2)) == 2


def test_similar_vibe_leaves_out_the_personas_shelved_works(mini_catalog, db: sqlite3.Connection):
    work = browse.get_work(db, SHINING_FILM)
    assert work is not None
    db.execute(
        "INSERT INTO shelf_items (persona_id, work_id, shelf) VALUES (1, ?, 'want')",
        (HILL_HOUSE_BOOK,),
    )
    similar = browse.similar_vibe(db, work, persona_id=1, k=6)
    assert HILL_HOUSE_BOOK not in ids(s.work for s in similar)


def test_similar_vibe_is_empty_without_a_vector(mini_catalog, db: sqlite3.Connection):
    db.execute("DELETE FROM vec_vibes WHERE work_id = ?", (SHINING_FILM,))
    work = browse.get_work(db, SHINING_FILM)
    assert work is not None
    assert browse.similar_vibe(db, work) == []
    assert browse.work_detail(db, SHINING_FILM).has_vector is False  # type: ignore[union-attr]


def test_work_detail_bundles_everything(mini_catalog, db: sqlite3.Connection):
    detail = browse.work_detail(db, SHINING_FILM)
    assert detail is not None
    assert detail.work.id == SHINING_FILM and detail.card is not None
    assert [lw.work.id for lw in detail.linked] == [SHINING_BOOK]
    assert detail.similar and detail.has_vector


def test_work_detail_of_an_unknown_work_is_none(mini_catalog, db: sqlite3.Connection):
    assert browse.work_detail(db, 9999) is None


def test_works_by_ids(mini_catalog, db: sqlite3.Connection):
    assert browse.works_by_ids(db, []) == {}
    got = browse.works_by_ids(db, [1, 11, 9999])
    assert set(got) == {1, 11}


def test_count_vectors(mini_catalog, db: sqlite3.Connection):
    assert browse.count_vectors(db) == 20
    db.execute("DELETE FROM vec_vibes")
    assert browse.count_vectors(db) == 0
