"""Shared Streamlit rendering. UI only: all data access goes through median.db helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

from median.db import connect, ensure_db, list_personas


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
