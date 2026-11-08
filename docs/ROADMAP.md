# Roadmap

Milestones in dependency order. Agents: tick boxes as you finish, and don't start a milestone until
the one before it is done (except where noted). Owners are placeholders. The team assigns them.

Rough timing for a ~3-week build: M0–M1 in week 1, M2–M3 in week 2, M4–M5 in week 3.

## M0 — Foundation
Owner: _TBD_
- [x] `pyproject.toml` with deps: streamlit, anthropic, openai, sqlite-vec, pydantic, httpx,
      tenacity, rapidfuzz, python-dotenv, SPARQLWrapper; dev: pytest, ruff
- [x] `median/config.py` (settings, model IDs, weights, price table)
- [x] `db/schema.sql` + `median/db.py` (`connect()` loads sqlite-vec; `init_db()`)
- [x] `median/llm.py` with usage logging and the `DAILY_USD_LIMIT` guard
- [x] Test fixtures: tiny catalog of ~20 works with vibe cards and fake vectors
- [x] Streamlit skeleton: Home + sidebar persona picker + empty pages
- [ ] Every teammate runs `uv sync` and `streamlit run` successfully

## M1 — Sample catalog (can start alongside the M0 UI work)
Owner: _TBD_

Status: the code, tests and the non-LLM half of the data build are done. Finishing needs two things
only people can do: **add Anthropic credits** (the account returned "credit balance is too low") and
**review the 50 vibe cards as a team**. Then run `uv run python -m pipeline.build --sample` once
(about $0.20) to finish `link_check`, `vibe_cards`, `embed`, `index` and `report`.

- [x] `pipeline/sources/tmdb.py` with response cache (search, details, discover, year-sliced discover)
- [x] `pipeline/sources/openlibrary.py` streaming dump reader (`ol_dumps.py`: parallel resumable
      download, streaming readers, record caches)
- [x] `pipeline/sources/wikidata.py` adaptation + series query (5,759 film/book pairs)
- [ ] `pipeline/enrich/vibe_cards.py`: settle the VibeCard template on 50 works and review as a team
      (generator, batch mode and review file are built and tested; **open** until credits are added,
      `uv run python -m pipeline.enrich.vibe_cards --run` has produced `data/vibe_review.md`, and the
      team has read it)
- [ ] `pipeline/enrich/embed.py` + `index` step (built and tested offline; runs in the finishing build)
- [ ] `pipeline.build --sample` produces a working `data/median.db`
      (done through `match`: 997 movies, 1,000 books, 177 verified adaptation pairs, 62 candidates
      for `link_check`; remaining: `link_check`, `vibe_cards`, `embed`, `index`, `report`)
- [ ] Gold eval files: `tests/eval/gold_adaptations.csv` (59 pairs, 38 in Wikidata and 21 not; all
      present in the sample) is done. `tests/eval/gold_vibes.csv` (32 pairs) is **proposed only**:
      the team must agree the pairs, then flip `status` to `agreed`.
- [ ] `pipeline/enrich/match.py` and `link_check.py` (ARCHITECTURE section 7 steps 4-5; added to M1 so
      M2's adaptation recall target is reachable). `match` has run on the real catalog and recovered
      18 of the 21 gold pairs Wikidata lacks; `link_check` is built and tested offline and runs in the
      finishing build.

## M2 — Overlap engine
Owner: _TBD_
- [ ] `median/recommend.py`: `overlap()` and `for_persona()` with reasons
- [ ] `median/taste.py` + persona seed data (`db/seed_personas.sql`)
- [ ] Unit tests for scoring, filters, diversity, taste math
- [ ] Eval passes: adaptation recall@10 ≥ 0.9, vibe recall@10 ≥ 0.5 on the sample catalog

## M3 — Core UI
Owner: _TBD_
- [ ] Browse page (search + filters)
- [ ] Work page with **Linked** and **Similar vibe** sections
- [ ] Shelves and 10-point rating widget (persona-scoped)
- [ ] Taste page
- [ ] Credits page with TMDB attribution

## M4 — Chat agent
Owner: _TBD_
- [ ] `median/agent.py`: tool loop, six tools, `present_recommendations` validation
- [ ] Chat page with streaming and recommendation cards
- [ ] 👍/👎 feedback updates the taste vector
- [ ] "Why?" explanations with Sonnet + cache
- [ ] Guardrail tests with recorded LLM responses

## M5 — Full catalog and demo
Owner: _TBD_
- [ ] Run the full `pipeline.build` once (~$5); check the `report` output
- [ ] Upload `median.db` to the team drive; everyone switches to it
- [ ] Rerun the eval on the full catalog; tune weights; record final numbers
- [ ] Write the demo script (3–4 scripted questions, one per persona)
- [ ] Pre-run the scripted questions so explanations are cached
- [ ] Record a backup screen capture of the full demo
- [ ] Feature freeze 3 days before the demo; rehearse on the demo laptop

## Stretch (only after M5)
- [ ] TV shows from TMDB (`media_type='tv'`)
- [ ] Review-based vibe signals
- [ ] Music (needs a new data source)
