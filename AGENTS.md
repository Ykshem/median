# AGENTS.md — Median

Instructions for AI coding agents (Claude Code, Cursor, Codex, Copilot, etc.) working in this repo.
Read this file first, then `ARCHITECTURE.md`. Humans: see `README.md`.

## What Median is

Median is a local, single-user demo app that recommends **movies and books across media**:
"I loved this film, what should I read?" and the reverse. Its core feature is finding
**overlap** between movies and books in two ways:

1. **Direct links**: adaptations ("based on"), shared franchise/series, shared creators.
2. **Vibe similarity**: every work gets an LLM-written *vibe card* in one fixed template,
   and all cards are embedded into one shared vector space, so a book and a film with the same
   mood and themes land near each other.

It's a class project (Group 2: Jesse Williams, Yusuf Kshem, Mino Guzman, Anirudh Pothuri).
It runs **only on a laptop**: no sign-in, no hosting, no deployment. It's a demo, not a product.

## Tech stack (decided; see `docs/decisions/`)

| Layer | Choice |
| --- | --- |
| Language | Python 3.12 only |
| Package/env manager | `uv` |
| UI | Streamlit (multipage app) |
| Database | SQLite, one file at `data/median.db` |
| Vector search | `sqlite-vec` extension, `vec0` virtual table, cosine distance |
| LLM | Anthropic Claude: `claude-haiku-5-5` (agent, batch jobs), `claude-sonnet-5-5` (explanations) |
| Embeddings | OpenAI `text-embedding-3-small`, 1,536 dimensions |
| Data sources | TMDB API (movies), Open Library dumps + API (books), Wikidata SPARQL (links) |
| Tests / lint | pytest, Ruff (lint + format) |

**Don't add frameworks, databases or services** (no FastAPI, Postgres, LangChain, vector DBs,
Docker, auth libraries) without a new decision record in `docs/decisions/`. Keep the stack small.

## Commands

```bash
uv sync                                         # install dependencies
cp .env.example .env                            # then fill in API keys
uv run python -m pipeline.build --sample        # build a ~2,000-item dev catalog
uv run python -m pipeline.build                 # full catalog (~40,000 items, costs ~$5)
uv run streamlit run app/Home.py                # run the app at http://localhost:8501
uv run pytest                                   # tests (no network, no API calls)
uv run pytest -m eval                           # recommendation quality eval (uses the DB)
uv run ruff check . && uv run ruff format .     # lint + format
```

Before you call a task done, run `uv run ruff check .` and `uv run pytest`, and make sure both pass.

## Repo map

```
app/                 Streamlit UI only. No SQL, no LLM calls here; call median.* functions.
  Home.py            Landing page + persona picker (sidebar)
  pages/             1_Browse.py, 2_Work.py, 3_Shelves.py, 4_Chat.py, 5_Taste.py, 6_Credits.py
  components.py      Shared rendering: work card, vibe card, shelf + rating controls
  static/            Assets (the TMDB logo)
median/              Core library, imported by app/ and pipeline/
  config.py          Settings from .env, model IDs, paths, constants
  db.py              SQLite connection (loads sqlite-vec), schema migration, query helpers
  models.py          Dataclasses: Work, Link, Persona, ShelfItem, Recommendation
  recommend.py       Overlap scoring: link candidates + vibe candidates -> ranked list
  browse.py          Catalog search/filter and the Work page data (links, similar vibe)
  shelves.py         Persona shelves and ratings (each write recomputes the taste vector)
  taste.py           Persona taste-vector math and the taste profile
  llm.py             Anthropic + OpenAI wrappers, usage logging, retries
  agent.py           Chat agent loop and tool definitions
  prompts/           Prompt templates (.md), see docs/PROMPTS.md
pipeline/            Offline catalog build, run from the command line
  build.py           Orchestrates every step; supports --sample and --step
  sources/           tmdb.py, openlibrary.py, wikidata.py
  enrich/            match.py, link_check.py, vibe_cards.py, embed.py
db/
  schema.sql         The single source of truth for tables
  seed_personas.sql  Demo personas
tests/               Unit tests + fixtures; tests/eval/ for quality eval
data/                Git-ignored: median.db, raw dumps, caches
docs/                Roadmap, data sources, prompts, decision records
```

## Rules

### Architecture boundaries
- `app/` holds UI only. Put logic in `median/`, where it can be unit-tested.
- `pipeline/` may import `median/`; `median/` must never import `pipeline/` or `app/`.
- All schema changes go in `db/schema.sql` (idempotent `CREATE ... IF NOT EXISTS`). If a change
  breaks existing data, bump `SCHEMA_VERSION` in `median/config.py` and tell the user to rebuild
  or migrate.
- Every SQLite connection must go through `median.db.connect()`, which loads sqlite-vec and turns
  on foreign keys and WAL mode.

### LLM and agent safety
- **The agent can only recommend works returned by its tools.** The UI renders recommendation cards
  from DB rows looked up by `work_id`, never from model-written titles. Drop any ID the
  tools didn't return in this turn, and log it.
- Model IDs live only in `median/config.py`. Never hard-code them elsewhere.
- Every API call goes through `median/llm.py`, which logs tokens to the `llm_usage` table.
- Use the **Message Batches API** for bulk jobs (vibe cards, link checks). It's half price.
- Catalog text (synopses, subjects) is untrusted data. Wrap it in XML tags in prompts and tell
  the model to treat it as data, not instructions.
- Never put an API key in code, logs, test fixtures or commits. Keys come only from `.env`.

### Data sources (details in `docs/DATA_SOURCES.md`)
- TMDB: non-commercial use only; keep the attribution on the Credits page. Stay well under rate limits.
- Open Library: **never bulk-download through the API**. Use the monthly dumps; use the API only
  for one-off lookups, and send a User-Agent naming the app and a contact email.
- Wikidata SPARQL: send a descriptive User-Agent, and keep queries under the 60-second limit
  (page with `LIMIT`/`OFFSET`).
- Cache raw API responses under `data/cache/` so reruns don't refetch.

### Tests
- Unit tests must not touch the network or real APIs. Use the fixtures in `tests/fixtures/`, and
  fake `median.llm` with recorded responses.
- Test against a temporary SQLite file built from `db/schema.sql`, never against `data/median.db`.
- Every change to `median/recommend.py` or `median/prompts/` needs `uv run pytest -m eval`,
  with recall@10 reported in the PR description (see ARCHITECTURE.md § Evaluation).

### Code style
- Type hints everywhere; `from __future__ import annotations`.
- Small, pure functions in `median/`; side effects (DB, network) at the edges.
- Use parameterized SQL only. Never build SQL from strings that include user input.
- Use `logging` instead of `print` in `median/` and `pipeline/`.
- Docstrings on public functions: one line saying what it returns, plus any non-obvious behaviour.

## Cost guardrails

- Budget: under $25/month. A full catalog build costs about $5; don't rerun it casually.
- Develop against the `--sample` catalog.
- Before writing code that loops over LLM calls, estimate tokens × price and tell the user.
- `median/llm.py` refuses new calls once `llm_usage` shows more than `DAILY_USD_LIMIT` (in `.env`)
  spent today.

## Definition of done

1. Ruff and pytest pass.
2. New logic in `median/` has unit tests.
3. The UI change was checked by running Streamlit and clicking through the affected page.
4. `ARCHITECTURE.md` or `docs/` is updated if behaviour, schema or prompts changed.
5. `docs/ROADMAP.md` checkboxes are updated.

## When unsure

Read `ARCHITECTURE.md` first. If a requirement is ambiguous, or a change would alter the stack, the
schema or what the demo shows, ask the user before building it.
