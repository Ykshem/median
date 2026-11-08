"""Credits and data-source attribution."""

from __future__ import annotations

import streamlit as st

from app.components import page_setup

page_setup("Credits", "🙏")

# TODO(M3): add the approved TMDB logo, less prominent than Median's own branding (DATA_SOURCES.md).
st.write("This product uses the TMDB API but is not endorsed or certified by TMDB.")
st.write("Book data comes from [Open Library](https://openlibrary.org/) (CC0 data dumps).")
st.write("Links between works come from [Wikidata](https://www.wikidata.org/) (CC0).")
st.caption(
    "Median is a class project by Group 2: Jesse Williams, Yusuf Kshem, Mino Guzman "
    "and Anirudh Pothuri."
)
