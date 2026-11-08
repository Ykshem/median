# ADR 0001 — Local Python stack: Streamlit + SQLite/sqlite-vec

- **Status:** Accepted
- **Date:** 2026-11-08
- **Deciders:** Median team (Group 2)

## Context

Median is a class demo due this term. It runs on one laptop: no sign-in, no hosting. The team
knows Python and TypeScript; the budget is under $25/month. The core feature is cross-media overlap
(direct links + vibe similarity), which needs relational data and vector search. Time is the
tightest constraint.

## Options considered

1. **Next.js + Postgres/pgvector in Docker:** a polished web UI and a clean path to going online,
   but two languages, Docker on every laptop, and more setup time.
2. **Streamlit + SQLite + sqlite-vec (chosen):** one language, one process, one database file, and a
   chat UI built in.
3. **Streamlit + a separate vector store (Chroma/Qdrant):** a second store to keep in sync with the
   relational data, with no benefit at ~40k items.

## Decision

Use Python 3.12, Streamlit, SQLite with the sqlite-vec extension (`vec0`, cosine distance), Claude
Haiku 5.5 / Sonnet 5.5 and OpenAI `text-embedding-3-small`. Use demo personas instead of accounts.

## Consequences

- ✅ Fastest path to a working demo; the whole catalog is one file the team can share.
- ✅ Links, shelves and vectors can be combined in a single SQL query.
- ⚠️ sqlite-vec is a 0.1.x (pre-1.0) release. Pin its version and cover KNN with a test.
- ⚠️ Streamlit looks like a data app and has limited layout control. Accepted for a demo.
- ⚠️ Some system Pythons can't load SQLite extensions. Use uv-managed Python.
- ↪️ Going online later means moving to Postgres + pgvector and a web frontend. The schema and
  scoring logic carry over; the UI is rewritten.

## Revisit when

Median needs multiple simultaneous users, hosting, or more than ~1M vectors.
