"""Work detail: vibe card, linked works, similar vibes in the other medium, shelf and rating."""

from __future__ import annotations

import streamlit as st

from app.components import (
    BROWSE_PAGE,
    db_conn,
    page_setup,
    render_cover,
    render_shelf_controls,
    render_vibe_card,
    render_work_card,
    work_link,
    work_meta,
)
from median import browse
from median.browse import SearchFilters

persona_id = page_setup("Work", "📖")

raw_id = st.query_params.get("work_id", "")
if not raw_id.isdigit():
    st.info("Pick a work to see its details. Search for one here, or open it from Browse.")
    quick = st.text_input("Find a work", placeholder="title or creator")
    if quick.strip():
        with db_conn() as conn:
            found = browse.search_works(conn, SearchFilters(query=quick.strip()), limit=10)
        if not found:
            st.warning("No works match.")
        for work in found:
            work_link(work)
    st.page_link(BROWSE_PAGE, label="Browse the catalog", icon="🔎")
    st.stop()

with db_conn() as conn:
    detail = browse.work_detail(conn, int(raw_id), persona_id)

if detail is None:
    st.error(f"There is no work with id {raw_id}.")
    st.page_link(BROWSE_PAGE, label="Back to Browse", icon="🔎")
    st.stop()

work = detail.work
head_cover, head_body = st.columns([1, 3])
with head_cover:
    render_cover(work, width=180)
with head_body:
    st.header(work.title)
    st.caption(work_meta(work))
    if work.synopsis:
        with st.expander("Synopsis", expanded=False):
            st.write(work.synopsis)
    if work.subjects:
        st.caption("Subjects: " + ", ".join(work.subjects[:8]))
    if persona_id is not None:
        render_shelf_controls(persona_id, work.id, key="work")

st.subheader("Vibe")
if detail.card is None:
    st.caption("This work has no vibe card yet.")
else:
    render_vibe_card(detail.card)

st.subheader("Linked")
if not detail.linked:
    st.caption("No verified adaptations, series or shared creators for this work.")
for lw in detail.linked:
    render_work_card(lw.work, key=f"linked-{lw.work.id}", relation=lw.relation)

other = {"movie": "books", "book": "movies"}.get(work.media_type, "works")
st.subheader(f"Similar vibe ({other})")
if not detail.has_vector:
    st.info(
        "Vibe similarity needs this work's embedding, which hasn't been built yet "
        "(`uv run python -m pipeline.build --sample --step embed`)."
    )
elif not detail.similar:
    st.caption("No similar works found.")
for sim in detail.similar:
    render_work_card(
        sim.work,
        key=f"similar-{sim.work.id}",
        reasons=sim.recommendation.reasons,
        persona_id=persona_id,
        controls=False,
    )
