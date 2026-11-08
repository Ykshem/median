"""Shared fixtures. Tests never touch data/median.db, the network or real API keys."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from median import config
from median.db import init_db
from tests.fixtures.catalog import load_mini_catalog


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point settings at a temp DB and dummy keys. load_dotenv() never overrides set variables,
    so the real `.env` cannot leak real keys into a test run."""
    monkeypatch.setenv("MEDIAN_DB_PATH", str(tmp_path / "median.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TMDB_API_READ_TOKEN", "test-tmdb-token")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.com")
    monkeypatch.setenv("DAILY_USD_LIMIT", "2")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A fresh temp-file DB built from db/schema.sql, with the demo personas seeded."""
    conn = init_db(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def mini_catalog(db: sqlite3.Connection) -> dict[str, Any]:
    """The 20-work mini catalog (cards, fake vectors, links) loaded into `db`; returns its JSON."""
    return load_mini_catalog(db)
