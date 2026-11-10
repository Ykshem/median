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

Status: the sample catalog is **built**: 997 movies, 1,000 books, 1,993 vibe cards (4 works have no
card), 1,993 embeddings and 224 verified adaptation pairs. `pipeline.build --sample --step report`
says all catalog targets are met. Two items stay open because only people can do them: **the team
reading the 50 vibe cards** (`data/vibe_review.md`) and **agreeing the 32 proposed gold vibe pairs**
(flip `status` to `agreed` in `tests/eval/gold_vibes.csv`). The vibe-recall number in M2 is measured
on those still-proposed pairs.

- [x] `pipeline/sources/tmdb.py` with response cache (search, details, discover, year-sliced discover)
- [x] `pipeline/sources/openlibrary.py` streaming dump reader (`ol_dumps.py`: parallel resumable
      download, streaming readers, record caches)
- [x] `pipeline/sources/wikidata.py` adaptation + series query (5,759 film/book pairs)
- [ ] `pipeline/enrich/vibe_cards.py`: settle the VibeCard template on 50 works and review as a team
      (generator, batch mode and review file are built, tested and have run; **open** until the team
      has read `data/vibe_review.md` and signed off the template)
- [x] `pipeline/enrich/embed.py` + `index` step (ran on the sample: 1,993 vectors, popularity
      percentiles, FTS index, taste vectors for all four personas; cost about $0.007)
- [x] `pipeline.build --sample` produces a working `data/median.db`
- [ ] Gold eval files: `tests/eval/gold_adaptations.csv` (59 pairs, 38 in Wikidata and 21 not; all
      present in the sample) is done. `tests/eval/gold_vibes.csv` (32 pairs) is **proposed only**:
      the team must agree the pairs, then flip `status` to `agreed`.
- [x] `pipeline/enrich/match.py` and `link_check.py` (ARCHITECTURE section 7 steps 4-5; added to M1 so
      M2's adaptation recall target is reachable). Both have run on the real catalog: `match`
      recovered 18 of the 21 gold pairs Wikidata lacks, `link_check` left no unverified links.

## M2 — Overlap engine
Owner: _TBD_

Status: done on the sample catalog. Eval (`uv run pytest -m eval`): adaptation recall@10 **1.000**
(59/59, target 0.9), vibe recall@10 **0.531** (17/32, target 0.5), medium mixing 0.30. The vibe number
has a one-pair margin and is measured on proposed (not yet team-agreed) gold pairs, so re-check it
after the team signs off the pairs and again on the full catalog (M5). It reached the target by
cutting the popularity weight from 0.05 to 0.002: see the note in `median/config.py`.

- [x] `median/recommend.py`: `overlap()` and `for_persona()` with reasons
- [x] `median/taste.py` + persona seed data (`db/seed_personas.sql`)
- [x] Unit tests for scoring, filters, diversity, taste math
- [x] Eval passes: adaptation recall@10 ≥ 0.9, vibe recall@10 ≥ 0.5 on the sample catalog

## M3 — Core UI
Owner: _TBD_

Status: done. All five pages are built, tested (page tests drive the real widgets against the mini
catalog) and were clicked through against a copy of the real 1,997-work catalog, including Similar
vibe and the Taste profile for all four personas.

- [x] Browse page (search + filters): title/creator search, medium, year, mood, sort, paging
- [x] Work page with **Linked** and **Similar vibe** sections
- [x] Shelves and 10-point rating widget (persona-scoped; every write recomputes the taste vector)
- [x] Taste page
- [x] Credits page with TMDB attribution (official TMDB logo in `app/static/`)

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
