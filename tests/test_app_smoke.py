"""Headless smoke test: every page renders against a temp DB with no exception."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from median.db import connect, ensure_db

REPO = Path(__file__).resolve().parent.parent
PAGES = sorted(str(p.relative_to(REPO)) for p in (REPO / "app" / "pages").glob("*.py"))


def _run(script: str) -> AppTest:
    at = AppTest.from_file(str(REPO / script), default_timeout=30)
    return at.run()


def test_there_are_six_pages() -> None:
    assert len(PAGES) == 6


@pytest.mark.parametrize("script", ["app/Home.py", *PAGES])
def test_page_renders_with_persona_picker(script: str) -> None:
    at = _run(script)
    assert not at.exception, [e.value for e in at.exception]
    picker = at.sidebar.selectbox[0]
    assert picker.label == "Who's browsing?"
    assert len(picker.options) == 4
    assert at.title  # every page shows a title


def test_first_run_creates_the_database() -> None:
    from median.config import get_settings

    path = get_settings().db_path
    assert not path.exists()
    _run("app/Home.py")
    assert path.exists()


def test_home_says_catalog_is_empty_without_a_catalog() -> None:
    at = _run("app/Home.py")
    assert any("catalog is empty" in i.value for i in at.info)


def test_persona_choice_is_kept_in_session_state() -> None:
    at = _run("app/Home.py")
    at.sidebar.selectbox[0].select(3).run()
    assert at.session_state["persona_id"] == 3
    assert at.sidebar.selectbox[0].value == 3


def test_credits_page_has_tmdb_disclaimer() -> None:
    at = _run("app/pages/6_Credits.py")
    assert any("not endorsed or certified by TMDB" in m.value for m in at.markdown)


def test_home_shows_catalog_size_when_works_exist() -> None:
    from median.config import get_settings
    from tests.fixtures.catalog import load_mini_catalog

    ensure_db()
    conn = connect(get_settings().db_path)
    load_mini_catalog(conn)
    conn.close()
    at = _run("app/Home.py")
    assert any("20 works" in c.value for c in at.caption)
