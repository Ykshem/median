"""Median landing page."""

from __future__ import annotations

import streamlit as st

from app.components import BROWSE_PAGE, db_conn, page_setup
from median.db import count_works

page_setup("Median")

st.write(
    "Tell Median a film you loved and it finds books with the same feel (and the novel it was "
    "adapted from). Tell it a book and it finds films. It combines **direct links** such as "
    "adaptations and series with **vibe similarity** across one shared space for movies and books."
)

with db_conn() as conn:
    works = count_works(conn)

if works == 0:
    st.info(
        "The catalog is empty. Copy the team's `median.db` into `data/`, or build a small one "
        "with `uv run python -m pipeline.build --sample`."
    )
else:
    st.caption(f"Catalog: {works:,} works.")

    with st.form("quick_search", border=False):
        text = st.text_input("Quick search", placeholder="A title or creator, e.g. the shining")
        if st.form_submit_button("Search") and text.strip():
            st.switch_page(BROWSE_PAGE, query_params={"q": text.strip()})
