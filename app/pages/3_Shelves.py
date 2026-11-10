"""The persona's shelves: want / in progress / finished / favorites, with ratings."""

from __future__ import annotations

import streamlit as st

from app.components import BROWSE_PAGE, db_conn, page_setup, render_work_card
from median import shelves

persona_id = page_setup("Shelves", "📚")
if persona_id is None:
    st.stop()

with db_conn() as conn:
    counts = shelves.shelf_counts(conn, persona_id)
    entries = {name: shelves.list_shelf(conn, persona_id, name) for name in shelves.SHELVES}

if sum(counts.values()) == 0:
    st.info("Nothing on this persona's shelves yet. Find something to add in Browse.")
    st.page_link(BROWSE_PAGE, label="Browse the catalog", icon="🔎")
    st.stop()

tabs = st.tabs([f"{shelves.SHELF_LABELS[name]} ({counts[name]})" for name in shelves.SHELVES])
for tab, name in zip(tabs, shelves.SHELVES, strict=True):
    with tab:
        if not entries[name]:
            st.caption("Empty.")
        for entry in entries[name]:
            render_work_card(
                entry.work,
                key=f"shelf-{name}-{entry.work.id}",
                persona_id=persona_id,
                controls=True,
            )
