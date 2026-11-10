"""Shared Streamlit rendering. UI only: all data access goes through median.db helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from median import shelves
from median.db import connect, ensure_db, list_personas
from median.models import VibeCard, Work

STATIC_DIR = Path(__file__).parent / "static"
MEDIA_ICON = {"movie": "🎬", "book": "📖"}
WORK_PAGE = "pages/2_Work.py"
BROWSE_PAGE = "pages/1_Browse.py"
NOT_SHELVED = "-"


@contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    """Open a connection for this rerun (creating an empty DB on first run) and close it after.

    Streamlit runs each rerun in its own thread, so connections are never shared between reruns.
    """
    conn = connect(ensure_db())
    try:
        yield conn
    finally:
        conn.close()


def render_persona_picker() -> int | None:
    """Render the sidebar persona picker and return the chosen persona id (None if none exist).

    The choice lives in `st.session_state["persona_id"]` so it survives page changes.
    """
    with db_conn() as conn:
        personas = list_personas(conn)
    st.sidebar.header("Persona")
    if not personas:
        st.sidebar.info("No personas found. Delete data/median.db and reload to recreate it.")
        return None
    ids = [p.id for p in personas]
    by_id = {p.id: p for p in personas}
    current = st.session_state.get("persona_id")
    index = ids.index(current) if current in ids else 0
    chosen = st.sidebar.selectbox(
        "Who's browsing?", ids, index=index, format_func=lambda i: by_id[i].name
    )
    st.session_state["persona_id"] = chosen
    if by_id[chosen].description:
        st.sidebar.caption(by_id[chosen].description)
    return chosen


def page_setup(title: str, icon: str = "🎬") -> int | None:
    """Configure the page, render the sidebar picker and the page title; return the persona id.

    Call this first on every page (`st.set_page_config` must be the first Streamlit command).
    """
    st.set_page_config(page_title=f"{title} · Median", page_icon=icon)
    persona_id = render_persona_picker()
    st.title(title)
    return persona_id


# Work rendering -------------------------------------------------------------------------------


def work_link(work: Work, label: str | None = None) -> None:
    """Render a link that opens the Work page for `work`."""
    default = f"{work.title} ({work.year})" if work.year else work.title
    st.page_link(
        WORK_PAGE,
        label=label or default,
        icon=MEDIA_ICON.get(work.media_type),
        query_params={"work_id": str(work.id)},
    )


def work_meta(work: Work) -> str:
    """Return a one-line summary: medium, year and creators."""
    parts = [work.media_type.title()]
    if work.year:
        parts.append(str(work.year))
    if work.creators:
        parts.append(", ".join(work.creators[:2]))
    return " · ".join(parts)


def render_cover(work: Work, width: int = 110) -> None:
    """Show the cover image, or a placeholder icon when there isn't one."""
    if work.cover_url:
        st.image(work.cover_url, width=width)
    else:
        st.markdown(f"## {MEDIA_ICON.get(work.media_type, '')}")


def render_work_card(
    work: Work,
    *,
    key: str,
    persona_id: int | None = None,
    relation: str | None = None,
    reasons: list[str] | None = None,
    controls: bool = False,
) -> None:
    """Render one work as a bordered card: cover, title link, meta, optional reasons and controls.

    `key` must be unique on the page. `relation` is a label like "Adapted from"; `reasons` are the
    scorer's machine-made reasons.
    """
    with st.container(border=True):
        cover, body = st.columns([1, 3])
        with cover:
            render_cover(work, width=80)
        with body:
            if relation:
                st.caption(relation)
            work_link(work, label=work.title)
            st.caption(work_meta(work))
            for reason in reasons or []:
                st.caption(f"• {reason}")
        if controls and persona_id is not None:
            render_shelf_controls(persona_id, work.id, key=key)


def render_vibe_card(card: VibeCard) -> None:
    """Render a vibe card: the logline, themes and moods as chips, then the other fields."""
    st.markdown(f"_{card.logline}_")
    st.markdown("**Themes:** " + " ".join(f":blue-badge[{t}]" for t in card.themes))
    st.markdown("**Mood:** " + " ".join(f":orange-badge[{m}]" for m in card.mood))
    left, right = st.columns(2)
    left.markdown(f"**Tone:** {card.tone}")
    left.markdown(f"**Setting:** {card.setting}")
    right.markdown(
        f"**Pacing:** {card.pacing} · **Intensity:** {card.intensity} · "
        f"**Audience:** {card.audience.replace('_', ' ')}"
    )
    right.markdown("**Hooks:** " + "; ".join(card.hooks))


# Shelf and rating controls --------------------------------------------------------------------


def _keys(persona_id: int, work_id: int, key: str) -> tuple[str, str]:
    return f"shelf-{key}-{persona_id}-{work_id}", f"rating-{key}-{persona_id}-{work_id}"


def _sync_widgets(persona_id: int, work_id: int, key: str) -> None:
    """Set the two widgets' state from the DB (after a write that may have changed both)."""
    shelf_key, rating_key = _keys(persona_id, work_id, key)
    with db_conn() as conn:
        item = shelves.get_item(conn, persona_id, work_id)
    st.session_state[shelf_key] = item.shelf if item else NOT_SHELVED
    st.session_state[rating_key] = item.rating if item else None


def _on_shelf_change(persona_id: int, work_id: int, key: str) -> None:
    shelf_key, _ = _keys(persona_id, work_id, key)
    choice = st.session_state[shelf_key]
    with db_conn() as conn:
        if choice == NOT_SHELVED:
            shelves.remove(conn, persona_id, work_id)
        else:
            shelves.set_shelf(conn, persona_id, work_id, choice)
    _sync_widgets(persona_id, work_id, key)


def _on_rating_change(persona_id: int, work_id: int, key: str) -> None:
    _, rating_key = _keys(persona_id, work_id, key)
    with db_conn() as conn:
        shelves.set_rating(conn, persona_id, work_id, st.session_state[rating_key])
    _sync_widgets(persona_id, work_id, key)


def render_shelf_controls(persona_id: int, work_id: int, *, key: str) -> None:
    """Render the shelf picker and the 1-10 rating widget for this persona and work.

    Changes are written (and the taste vector recomputed) immediately. `key` must be unique on the
    page; the widgets' state is re-read from the DB whenever they first appear on a page.
    """
    shelf_key, rating_key = _keys(persona_id, work_id, key)
    if shelf_key not in st.session_state or rating_key not in st.session_state:
        _sync_widgets(persona_id, work_id, key)
    args = (persona_id, work_id, key)
    shelf_col, rating_col = st.columns([1, 2])
    shelf_col.selectbox(
        "Shelf",
        [NOT_SHELVED, *shelves.SHELVES],
        format_func=lambda s: "Not on a shelf" if s == NOT_SHELVED else shelves.SHELF_LABELS[s],
        key=shelf_key,
        on_change=_on_shelf_change,
        args=args,
    )
    rating_col.segmented_control(
        "Rating (1-10)",
        list(shelves.RATING_RANGE),
        key=rating_key,
        on_change=_on_rating_change,
        args=args,
    )
    rating = st.session_state.get(rating_key)
    if rating:
        rating_col.caption(f"{shelves.stars_text(rating)}  ({rating}/10)")
