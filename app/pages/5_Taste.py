"""Taste profile: what the persona's taste vector looks like, and their rating history."""

from __future__ import annotations

import streamlit as st

from app.components import BROWSE_PAGE, db_conn, page_setup, render_work_card
from median import browse, shelves, taste
from median.config import TASTE_MIN_SIGNALS

persona_id = page_setup("Taste", "🎯")
if persona_id is None:
    st.stop()

with db_conn() as conn:
    profile = taste.profile(conn, persona_id)
    rated = [e for e in shelves.list_shelf(conn, persona_id) if e.item.rating is not None]
    works = browse.works_by_ids(
        conn, [*profile.nearest[:6], *profile.liked[:5], *profile.disliked[:5]]
    )
    embeddings = browse.count_vectors(conn)

if embeddings == 0:
    st.warning(
        "The catalog has no vibe embeddings yet, so taste vectors can't be computed. "
        "Run `uv run python -m pipeline.build --sample --step embed` (needs OpenAI credits)."
    )
elif not profile.has_vector:
    st.info(
        f"A taste profile needs at least {TASTE_MIN_SIGNALS} ratings or favorites "
        f"(this persona has {profile.signals}). Rate some works to build one."
    )
    st.page_link(BROWSE_PAGE, label="Browse the catalog", icon="🔎")

if profile.has_vector:
    st.caption(f"Based on {profile.signals} ratings and favorites.")
    themes_col, moods_col = st.columns(2)
    with themes_col:
        st.subheader("Top themes")
        st.bar_chart(dict(profile.themes), horizontal=True)
    with moods_col:
        st.subheader("Top moods")
        st.bar_chart(dict(profile.moods), horizontal=True)

    st.subheader("Closest to their taste")
    st.caption("The works whose vibe sits nearest the taste vector.")
    cols = st.columns(2)
    for i, work_id in enumerate(profile.nearest[:6]):
        if work_id in works:
            with cols[i % 2]:
                render_work_card(works[work_id], key=f"taste-near-{work_id}")

if profile.liked or profile.disliked:
    liked_col, disliked_col = st.columns(2)
    with liked_col:
        st.subheader("Pulls toward")
        for work_id in profile.liked[:5]:
            if work_id in works:
                st.markdown(f"- {works[work_id].title} ({works[work_id].media_type})")
    with disliked_col:
        st.subheader("Pushes away")
        if not profile.disliked:
            st.caption("Nothing disliked yet.")
        for work_id in profile.disliked[:5]:
            if work_id in works:
                st.markdown(f"- {works[work_id].title} ({works[work_id].media_type})")

st.subheader("Rating history")
if not rated:
    st.caption("No ratings yet.")
else:
    distribution = {str(n): 0 for n in shelves.RATING_RANGE}
    for entry in rated:
        distribution[str(entry.item.rating)] += 1
    st.bar_chart(distribution, x_label="Rating", y_label="Works")
    st.dataframe(
        [
            {
                "Title": e.work.title,
                "Medium": e.work.media_type,
                "Year": e.work.year,
                "Rating": f"{shelves.stars_text(e.item.rating)} ({e.item.rating}/10)",
                "Shelf": shelves.SHELF_LABELS[e.item.shelf],
            }
            for e in rated
        ],
        hide_index=True,
        width="stretch",
    )
