-- Median schema. Single source of truth; every statement is idempotent.
-- If a change breaks existing data, bump SCHEMA_VERSION in median/config.py.

-- Catalog -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS works (
  id          INTEGER PRIMARY KEY,
  media_type  TEXT NOT NULL CHECK (media_type IN ('movie','book','tv','music')),
  title       TEXT NOT NULL,
  year        INTEGER,
  creators    TEXT,            -- JSON array of names (directors/authors)
  synopsis    TEXT,
  subjects    TEXT,            -- JSON array: genres, TMDB keywords, OL subjects
  popularity  REAL,            -- 0..1 percentile within media_type
  cover_url   TEXT
);

CREATE TABLE IF NOT EXISTS external_ids (
  work_id     INTEGER NOT NULL REFERENCES works(id),
  source      TEXT NOT NULL CHECK (source IN ('tmdb','openlibrary','wikidata')),
  external_id TEXT NOT NULL,
  PRIMARY KEY (source, external_id)
);

CREATE TABLE IF NOT EXISTS work_links (
  from_work   INTEGER NOT NULL REFERENCES works(id),
  to_work     INTEGER NOT NULL REFERENCES works(id),
  link_type   TEXT NOT NULL CHECK (link_type IN ('adaptation_of','same_franchise','same_creator')),
  source      TEXT NOT NULL CHECK (source IN ('wikidata','tmdb_credits','title_match')),
  confidence  REAL NOT NULL DEFAULT 1.0,   -- 0..1
  verified    INTEGER NOT NULL DEFAULT 0,  -- 1 = passed LLM link check or from Wikidata
  PRIMARY KEY (from_work, to_work, link_type)
);
-- adaptation_of: from_work (movie) is adapted from to_work (book)

CREATE TABLE IF NOT EXISTS vibe_cards (
  work_id        INTEGER PRIMARY KEY REFERENCES works(id),
  card_json      TEXT NOT NULL,    -- validated VibeCard (ARCHITECTURE.md section 4)
  card_text      TEXT NOT NULL,    -- flattened text that was embedded
  model          TEXT NOT NULL,
  prompt_version TEXT NOT NULL
);

-- Vector index (sqlite-vec). media_type is a metadata column, so KNN can filter on it.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_vibes USING vec0(
  work_id    INTEGER PRIMARY KEY,
  embedding  float[1536] distance_metric=cosine,
  media_type TEXT
);

-- Keyword search over titles and creators (external content: call rebuild_fts after loading works)
CREATE VIRTUAL TABLE IF NOT EXISTS works_fts USING fts5(
  title, creators, content='works', content_rowid='id'
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
-- keys: schema_version, embedding_model, embedding_dims, catalog_built_at, vibe_prompt_version

-- Personas and their data --------------------------------------------
CREATE TABLE IF NOT EXISTS personas (
  id           INTEGER PRIMARY KEY,
  name         TEXT NOT NULL,
  description  TEXT,
  taste_vector BLOB,             -- float32[1536], NULL until >= 3 signals
  updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS shelf_items (
  persona_id INTEGER NOT NULL REFERENCES personas(id),
  work_id    INTEGER NOT NULL REFERENCES works(id),
  shelf      TEXT NOT NULL CHECK (shelf IN ('want','in_progress','finished','favorite')),
  rating     INTEGER CHECK (rating BETWEEN 1 AND 10),   -- 10-point = half-star UI
  note       TEXT,
  added_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (persona_id, work_id)
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id              INTEGER PRIMARY KEY,
  persona_id      INTEGER NOT NULL REFERENCES personas(id),
  session_id      TEXT NOT NULL,
  role            TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content         TEXT NOT NULL,
  recommended_ids TEXT,          -- JSON array of work ids shown as cards
  created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rec_feedback (
  id         INTEGER PRIMARY KEY,
  persona_id INTEGER NOT NULL REFERENCES personas(id),
  work_id    INTEGER NOT NULL REFERENCES works(id),
  seed_id    INTEGER REFERENCES works(id),
  signal     TEXT NOT NULL CHECK (signal IN ('up','down')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS explanations (       -- cache of Sonnet "Why?" answers
  persona_id INTEGER NOT NULL, work_id INTEGER NOT NULL,
  seed_id    INTEGER NOT NULL DEFAULT 0,   -- 0 = no seed work (taste-based)
  text       TEXT NOT NULL,
  PRIMARY KEY (persona_id, work_id, seed_id)
);

CREATE TABLE IF NOT EXISTS llm_usage (
  id INTEGER PRIMARY KEY, purpose TEXT, model TEXT,
  input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Lookup indexes ------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_work_links_to ON work_links(to_work);
CREATE INDEX IF NOT EXISTS idx_external_ids_work ON external_ids(work_id);
CREATE INDEX IF NOT EXISTS idx_shelf_items_work ON shelf_items(work_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_works_media ON works(media_type);
