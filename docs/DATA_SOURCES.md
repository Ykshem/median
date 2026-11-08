# Data sources

Terms and limits as checked on 2026-11-08. Recheck the linked pages before relying on a limit.

## TMDB (movies)

- **Docs:** https://developer.themoviedb.org/docs
- **Auth:** API Read Access Token (v4) as `Authorization: Bearer <token>` (`TMDB_API_READ_TOKEN`).
- **Terms:** free for **non-commercial** use with attribution. Commercial use needs a separate
  written agreement. Median is a non-commercial class project.
- **Attribution (required):** the Credits page must show the TMDB logo (an approved one, less
  prominent than Median's branding) and: *"This product uses the TMDB API but is not endorsed or
  certified by TMDB."*
- **Rate limits:** stay at or below roughly 40 requests per 10 seconds, and back off on HTTP 429.
- **Usage in Median:** discover/popular lists to choose ~20k movies; `/movie/{id}` with
  `append_to_response=keywords,credits,external_ids`; poster URLs from the image CDN.
- **Caching:** cache every raw JSON response under `data/cache/tmdb/` so reruns are free and fast.

## Open Library (books)

- **API docs:** https://openlibrary.org/developers/api
- **Dumps:** https://openlibrary.org/developers/dumps (monthly; works, editions, authors, ratings and
  reading-log files; gzipped TSV with a JSON column). Data is public domain (CC0).
- **Rules:** the APIs are **not for bulk download**; use the dumps for the catalog. API limits are
  1 request/s by default, 3 requests/s with a User-Agent that names the app and gives a contact,
  e.g. `Median/0.1 (class project; contact: $CONTACT_EMAIL)`.
- **Usage in Median:** stream the works, authors and reading-log dumps; keep the top ~20k works by
  reading-log count that have a description or subjects. Covers come from
  `https://covers.openlibrary.org/b/id/{cover_id}-M.jpg` and are loaded by the browser on display.
  Don't bulk-download covers.
- **Data quality:** community-edited; expect duplicate works and inconsistent author names.
  Normalize names before matching.

## Wikidata (links between works)

- **Endpoint:** https://query.wikidata.org/sparql
- **Rules:** send a descriptive User-Agent (generic agents get 403/429). Queries time out at 60 s;
  page with `LIMIT`/`OFFSET`, and filter early.
- **Properties used** (check each on wikidata.org before relying on it): P144 *based on*,
  P179 *part of the series*, P50 *author*, P57 *director*, plus the external-ID properties
  for TMDB movie IDs and Open Library IDs.
- **Coverage warning:** one study found only about a third of English Wikipedia's "films based on
  novels" had a Wikidata *based on* link
  (https://github.com/tommasobattisti/TheSpaghettiJunction). That's why the pipeline adds TMDB
  credit matching and an LLM link check.

## LLM and embedding APIs

| Model | Use | Price (USD per million tokens, input / output) |
| --- | --- | --- |
| `claude-haiku-5-5` | Agent, vibe cards (batch), link checks (batch) | $0.10 / $0.50 for prompts up to 100K tokens; batch half price |
| `claude-sonnet-5-5` | "Why?" explanations | $2 / $10 |
| `text-embedding-3-small` | Vibe card + query embeddings | $0.02 input |

Claude prices are from a third-party summary dated 2026-10-29; confirm on Anthropic's pricing page and
keep `median/config.py`'s price table in sync. Expected spend: about $5 per full catalog build plus a few
dollars of chat for the whole term.
