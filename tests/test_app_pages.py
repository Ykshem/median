"""Behaviour of the M3 pages (Browse, Work, Shelves, Taste, Credits) against the mini catalog."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from median import shelves
from median.config import get_settings
from median.db import connect, ensure_db
from tests.fixtures.catalog import load_mini_catalog

REPO = Path(__file__).resolve().parent.parent
SHINING_FILM, SHINING_BOOK, HILL_HOUSE = 1, 11, 12
HORROR = 1  # the first persona, which the picker selects by default


def run(page: str, **query_params: str) -> AppTest:
    """Open the app at Home, then go to `page` (e.g. 'pages/2_Work.py'), like a real session."""
    at = AppTest.from_file(str(REPO / "app" / "Home.py"), default_timeout=30)
    at.query_params.update(query_params)
    at.run()
    if page != "Home":
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """A connection to the app's temp DB, loaded with the mini catalog."""
    ensure_db()
    c = connect(get_settings().db_path)
    load_mini_catalog(c)
    yield c
    c.close()


def texts(elements) -> list[str]:
    return [e.value for e in elements]


# Browse ---------------------------------------------------------------------------------------


def test_browse_on_an_empty_catalog_says_so() -> None:
    at = run("pages/1_Browse.py")
    assert any("catalog is empty" in v for v in texts(at.info))


def test_browse_lists_the_catalog(conn) -> None:
    at = run("pages/1_Browse.py")
    assert any(c.startswith("20 works · page 1 of 2") for c in texts(at.caption))
    assert len(at.page_link) == 18  # one page of cards


def test_browse_search_filters_results(conn) -> None:
    at = run("pages/1_Browse.py")
    at.text_input(key="browse_q").input("shining").run()
    assert not at.exception
    assert any(c.startswith("2 works") for c in texts(at.caption))


def test_browse_home_quick_search_arrives_as_a_query(conn) -> None:
    at = run("pages/1_Browse.py", q="orient express")
    assert at.text_input(key="browse_q").value == "orient express"
    assert any(c.startswith("2 works") for c in texts(at.caption))


def test_browse_with_no_matches_warns(conn) -> None:
    at = run("pages/1_Browse.py")
    at.text_input(key="browse_q").input("zzzzqqqq").run()
    assert any("No works match" in v for v in texts(at.warning))


def test_browse_medium_filter(conn) -> None:
    at = run("pages/1_Browse.py")
    at.selectbox[0].select("Books").run()  # at.selectbox leaves out the sidebar's persona picker
    assert not at.exception
    assert any(c.startswith("10 works") for c in texts(at.caption))


# Work -----------------------------------------------------------------------------------------


def test_work_page_without_an_id_offers_search(conn) -> None:
    at = run("pages/2_Work.py")
    assert any("Pick a work" in v for v in texts(at.info))
    at.text_input[0].input("hobbit").run()
    assert not at.exception
    assert at.page_link  # results are links


def test_work_page_with_an_unknown_id_errors(conn) -> None:
    at = run("pages/2_Work.py", work_id="9999")
    assert any("no work with id 9999" in v for v in texts(at.error))


def test_work_page_shows_vibe_linked_and_similar(conn) -> None:
    at = run("pages/2_Work.py", work_id=str(SHINING_FILM))
    assert texts(at.main.header) == ["The Shining"]
    assert texts(at.subheader) == ["Vibe", "Linked", "Similar vibe (books)"]
    assert any("Adapted from" in c for c in texts(at.caption))
    assert any("Shared themes" in c or "Fits" in c for c in texts(at.caption))


def test_work_page_explains_missing_embeddings(conn) -> None:
    conn.execute("DELETE FROM vec_vibes WHERE work_id = ?", (SHINING_FILM,))
    conn.commit()
    at = run("pages/2_Work.py", work_id=str(SHINING_FILM))
    assert any("hasn't been built yet" in v for v in texts(at.info))


def test_work_page_labels_a_same_creator_link(conn) -> None:
    at = run("pages/2_Work.py", work_id="20")  # The Andromeda Strain
    assert any("Same creator" in c for c in texts(at.caption))


def test_shelf_picker_writes_to_the_db(conn) -> None:
    at = run("pages/2_Work.py", work_id=str(SHINING_FILM))
    at.selectbox(key=f"shelf-work-{HORROR}-{SHINING_FILM}").select("want").run()
    assert not at.exception
    item = shelves.get_item(conn, HORROR, SHINING_FILM)
    assert item is not None and item.shelf == "want"
    at.selectbox(key=f"shelf-work-{HORROR}-{SHINING_FILM}").select("-").run()
    assert shelves.get_item(conn, HORROR, SHINING_FILM) is None


def test_rating_widget_writes_and_files_the_work_as_finished(conn) -> None:
    at = run("pages/2_Work.py", work_id=str(SHINING_FILM))
    at.segmented_control(key=f"rating-work-{HORROR}-{SHINING_FILM}").set_value(7).run()
    assert not at.exception
    item = shelves.get_item(conn, HORROR, SHINING_FILM)
    assert item is not None and (item.rating, item.shelf) == (7, "finished")
    # the shelf picker now reflects the automatic 'finished' shelf, and the stars show
    assert at.selectbox(key=f"shelf-work-{HORROR}-{SHINING_FILM}").value == "finished"
    assert any("★★★½☆" in c for c in texts(at.caption))


def test_removing_from_the_shelf_clears_the_rating_widget(conn) -> None:
    shelves.set_rating(conn, HORROR, SHINING_FILM, 9)
    at = run("pages/2_Work.py", work_id=str(SHINING_FILM))
    assert at.segmented_control(key=f"rating-work-{HORROR}-{SHINING_FILM}").value == 9
    at.selectbox(key=f"shelf-work-{HORROR}-{SHINING_FILM}").select("-").run()
    assert at.segmented_control(key=f"rating-work-{HORROR}-{SHINING_FILM}").value is None
    assert shelves.get_item(conn, HORROR, SHINING_FILM) is None


def test_ratings_are_scoped_to_the_selected_persona(conn) -> None:
    at = run("pages/2_Work.py", work_id=str(SHINING_FILM))
    at.sidebar.selectbox[0].select(2).run()  # switch to the cozy mystery persona
    at.segmented_control(key=f"rating-work-2-{SHINING_FILM}").set_value(3).run()
    assert shelves.get_item(conn, 2, SHINING_FILM).rating == 3  # type: ignore[union-attr]
    assert shelves.get_item(conn, HORROR, SHINING_FILM) is None


# Shelves --------------------------------------------------------------------------------------


def test_shelves_page_starts_empty_for_a_persona_without_items(conn) -> None:
    at = run("pages/3_Shelves.py")
    assert any("Nothing on this persona's shelves" in v for v in texts(at.info))


def test_shelves_page_shows_tabs_with_counts(conn) -> None:
    shelves.set_shelf(conn, HORROR, 1, "want")
    shelves.set_shelf(conn, HORROR, 2, "want")
    shelves.set_rating(conn, HORROR, 11, 9)
    at = run("pages/3_Shelves.py")
    assert [t.label for t in at.tabs] == [
        "Want (2)",
        "In progress (0)",
        "Finished (1)",
        "Favorites (0)",
    ]


def test_shelves_page_can_move_and_rate(conn) -> None:
    shelves.set_shelf(conn, HORROR, SHINING_FILM, "want")
    at = run("pages/3_Shelves.py")
    at.selectbox(key=f"shelf-shelf-want-{SHINING_FILM}-{HORROR}-{SHINING_FILM}").select(
        "favorite"
    ).run()
    assert shelves.get_item(conn, HORROR, SHINING_FILM).shelf == "favorite"  # type: ignore[union-attr]


# Taste ----------------------------------------------------------------------------------------


def test_taste_page_without_embeddings_explains_why() -> None:
    at = run("pages/5_Taste.py")
    assert any("no vibe embeddings" in v for v in texts(at.warning))


def test_taste_page_asks_for_more_ratings(conn) -> None:
    shelves.set_rating(conn, HORROR, 1, 10)
    at = run("pages/5_Taste.py")
    assert any("at least 3" in v for v in texts(at.info))
    assert "Rating history" in texts(at.subheader)  # still shown without a taste vector


def test_taste_page_shows_profile_once_there_is_a_vector(conn) -> None:
    for work_id in (1, 2, 11, 12):
        shelves.set_rating(conn, HORROR, work_id, 10)
    shelves.set_rating(conn, HORROR, 3, 1)
    at = run("pages/5_Taste.py")
    headings = texts(at.subheader)
    for expected in ("Top themes", "Top moods", "Closest to their taste", "Rating history"):
        assert expected in headings
    assert "Pulls toward" in headings and "Pushes away" in headings
    assert len(at.get("vega_lite_chart")) == 3  # themes, moods and the rating distribution


# Credits --------------------------------------------------------------------------------------


def test_credits_page_shows_the_tmdb_logo_and_disclaimer() -> None:
    at = run("pages/6_Credits.py")
    assert any("not endorsed or certified by TMDB" in m.value for m in at.markdown)
    assert at.get("image")
    logo = REPO / "app" / "static" / "tmdb_logo.svg"
    assert logo.read_text(encoding="utf-8").lstrip().startswith("<svg")
