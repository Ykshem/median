"""Settings, model IDs, scoring weights and the price table.

Everything environment-specific comes from `.env`. Model IDs live only in this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"
SEED_PERSONAS_PATH = REPO_ROOT / "db" / "seed_personas.sql"
PROMPTS_DIR = REPO_ROOT / "median" / "prompts"
CACHE_DIR = REPO_ROOT / "data" / "cache"  # raw API responses, so reruns don't refetch
REVIEW_PATH = REPO_ROOT / "data" / "vibe_review.md"
DUMPS_DIR = REPO_ROOT / "data" / "dumps"  # Open Library monthly dumps (~5 GB), git-ignored
SAMPLE_PLAN_PATH = REPO_ROOT / "data" / "sample_plan.json"
GOLD_DIR = REPO_ROOT / "tests" / "eval"
GOLD_ADAPTATIONS_PATH = GOLD_DIR / "gold_adaptations.csv"
GOLD_VIBES_PATH = GOLD_DIR / "gold_vibes.csv"

# Models ---------------------------------------------------------------------------------------
AGENT_MODEL = "claude-haiku-5-5"
EXPLAIN_MODEL = "claude-sonnet-5-5"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

# Versions: bump to force regeneration / rebuild --------------------------------------------------
VIBE_PROMPT_VERSION = "v1"
LINK_PROMPT_VERSION = "v1"
SCHEMA_VERSION = 2  # v2: works.popularity_raw

# Recommendation scoring (ARCHITECTURE.md section 5) ---------------------------------------------
SCORE_WEIGHTS = {"link": 0.45, "vibe": 0.35, "taste": 0.15, "popularity": 0.05}
LINK_STRENGTH = {"adaptation_of": 1.0, "same_franchise": 0.8, "same_creator": 0.5}

# Prices in USD per million tokens: (input, output). Batch requests cost BATCH_DISCOUNT x.
# Source: docs/DATA_SOURCES.md (third-party summary dated 2026-10-29). Keep in sync with the
# providers' pricing pages.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    AGENT_MODEL: (0.10, 0.50),
    EXPLAIN_MODEL: (2.00, 10.00),
    EMBED_MODEL: (0.02, 0.0),
}
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class Settings:
    """Values read from the environment. API keys are excluded from repr so they aren't logged."""

    anthropic_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)
    tmdb_api_read_token: str = field(default="", repr=False)
    contact_email: str = ""
    db_path: Path = REPO_ROOT / "data" / "median.db"
    daily_usd_limit: float = 2.0


def _resolve_db_path(raw: str | None) -> Path:
    """Return the DB path, resolving relative values against the repo root (not the CWD)."""
    path = Path(raw) if raw else Path("data") / "median.db"
    return path if path.is_absolute() else REPO_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings. Call `get_settings.cache_clear()` after changing the env."""
    load_dotenv(REPO_ROOT / ".env")
    env = os.environ
    return Settings(
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        tmdb_api_read_token=env.get("TMDB_API_READ_TOKEN", ""),
        contact_email=env.get("CONTACT_EMAIL", ""),
        db_path=_resolve_db_path(env.get("MEDIAN_DB_PATH")),
        daily_usd_limit=float(env.get("DAILY_USD_LIMIT") or 2.0),
    )
