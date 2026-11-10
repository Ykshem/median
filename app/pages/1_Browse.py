"""Browse and search the catalog: text search plus medium, year, mood and sort filters."""

from __future__ import annotations

import math

import streamlit as st

from app.components import db_conn, page_setup, render_work_card
from median import browse
from median.browse import SearchFilters

PAGE_SIZE = 18
COLUMNS = 3
SORT_LABELS = {
    "relevance": "Best match",
    "popular": "Most popular",
    "year_newest": "Newest first",
    "year_oldest": "Oldest first",
    "title": "Title A-Z",
}
MEDIA_LABELS = {"any": "Movies and books", "movie": "Movies", "book": "Books"}

persona_id = page_setup("Browse", "🔎")

with db_conn() as conn:
    moods = browse.list_moods(conn)
    bounds = browse.year_bounds(conn)

if bounds is None:
    st.info("The catalog is empty. Build one with `uv run python -m pipeline.build --sample`.")
    st.stop()

# Home's quick search arrives as ?q=...; seed the box once, then the widget owns its value.
if "browse_q" not in st.session_state:
    st.session_state["browse_q"] = st.query_params.get("q", "")

query = st.text_input("Search titles and creators", key="browse_q", placeholder="e.g. shining")
c1, c2, c3 = st.columns(3)
media = c1.selectbox("Medium", list(MEDIA_LABELS), format_func=MEDIA_LABELS.__getitem__)
mood = c2.selectbox("Mood", [None, *moods], format_func=lambda m: "Any mood" if m is None else m)
sort = c3.selectbox("Sort", list(SORT_LABELS), format_func=SORT_LABELS.__getitem__)
year_min, year_max = st.slider("Year", bounds[0], bounds[1], bounds, key="browse_years")

filters = SearchFilters(
    query=query.strip(),
    media_type=media,
    year_min=year_min if year_min > bounds[0] else None,
    year_max=year_max if year_max < bounds[1] else None,
    mood=mood,
    sort=sort,
)

# Go back to page 1 whenever any filter changes.
if st.session_state.get("browse_filters") != filters:
    st.session_state["browse_filters"] = filters
    st.session_state["browse_page"] = 1

with db_conn() as conn:
    total = browse.count_matches(conn, filters)
    pages = max(1, math.ceil(total / PAGE_SIZE))
    page = min(st.session_state.get("browse_page", 1), pages)
    works = browse.search_works(conn, filters, PAGE_SIZE, (page - 1) * PAGE_SIZE)

if total == 0:
    st.warning("No works match. Try fewer words or loosen a filter.")
    st.stop()

st.caption(f"{total:,} work{'s' if total != 1 else ''} · page {page} of {pages}")
cols = st.columns(COLUMNS)
for i, work in enumerate(works):
    with cols[i % COLUMNS]:
        render_work_card(work, key=f"browse-{work.id}", persona_id=persona_id)

if pages > 1:
    prev_col, mid_col, next_col = st.columns([1, 2, 1])
    if prev_col.button("← Previous", disabled=page <= 1):
        st.session_state["browse_page"] = page - 1
        st.rerun()
    mid_col.caption(f"Page {page} of {pages}")
    if next_col.button("Next →", disabled=page >= pages):
        st.session_state["browse_page"] = page + 1
        st.rerun()
