# Median

A local demo that recommends **movies and books across media**. Tell it a film you loved and it finds
books with the same feel (and the novel it was adapted from); tell it a book and it finds films.

Median combines **direct links** (adaptations, series, shared creators) with **vibe similarity**
(LLM-written, media-neutral "vibe cards" embedded in one vector space), and explains every pick
through a chat agent.

Class project, Group 2: Jesse Williams, Yusuf Kshem, Mino Guzman, Anirudh Pothuri.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/), plus API keys for Anthropic, OpenAI and TMDB.

```bash
git clone <repo-url> median && cd median
uv python install 3.12        # uv's Python can load SQLite extensions
uv sync
cp .env.example .env          # fill in your keys and contact email
```

Get a catalog, using **one** of these:

```bash
# A) Use the team's shared database (recommended): download median.db from the shared drive
#    into data/median.db

# B) Build a small dev catalog yourself (~2,000 works, a few cents of API calls)
uv run python -m pipeline.build --sample
```

Run the app:

```bash
uv run streamlit run app/Home.py      # opens http://localhost:8501
```

## Everyday commands

| Task | Command |
| --- | --- |
| Run tests | `uv run pytest` |
| Recommendation quality eval | `uv run pytest -m eval` |
| Lint and format | `uv run ruff check . && uv run ruff format .` |
| Rebuild one pipeline step | `uv run python -m pipeline.build --step vibe_cards` |
| Full catalog build (~$5) | `uv run python -m pipeline.build` |

## Docs

- `AGENTS.md`: rules and commands for AI coding agents (and a good summary for humans)
- `ARCHITECTURE.md`: how it works: schema, scoring, agent, pipeline
- `docs/ROADMAP.md`: milestones and task checklist
- `docs/DATA_SOURCES.md`: TMDB, Open Library, Wikidata terms and limits
- `docs/PROMPTS.md`: prompt specs for vibe cards, link checks, the agent and explanations
- `docs/decisions/`: architecture decision records

## Credits

This product uses the TMDB API but is not endorsed or certified by TMDB. Book data from
Open Library; links from Wikidata.
