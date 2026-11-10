"""Credits and data-source attribution."""

from __future__ import annotations

import streamlit as st

from app.components import STATIC_DIR, page_setup

page_setup("Credits", "🙏")

st.image(str(STATIC_DIR / "tmdb_logo.svg"), width=90, alt="The Movie Database (TMDB) logo")
st.write("This product uses the TMDB API but is not endorsed or certified by TMDB.")
st.write("Book data comes from [Open Library](https://openlibrary.org/) (CC0 data dumps).")
st.write("Links between works come from [Wikidata](https://www.wikidata.org/) (CC0).")
st.caption(
    "Median is a class project by Group 2: Jesse Williams, Yusuf Kshem, Mino Guzman "
    "and Anirudh Pothuri."
)
