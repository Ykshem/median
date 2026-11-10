from __future__ import annotations

from pathlib import Path

import pytest

from median import config


def test_env_overrides_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_USD_LIMIT", "7.5")
    config.get_settings.cache_clear()
    settings = config.get_settings()
    assert settings.daily_usd_limit == 7.5
    assert settings.anthropic_api_key == "test-anthropic-key"
    assert settings.contact_email == "test@example.com"


def test_daily_limit_defaults_to_two_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_USD_LIMIT", "")
    config.get_settings.cache_clear()
    assert config.get_settings().daily_usd_limit == 2.0


def test_relative_db_path_resolves_against_repo_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEDIAN_DB_PATH", "data/other.db")
    monkeypatch.chdir(tmp_path)
    config.get_settings.cache_clear()
    assert config.get_settings().db_path == config.REPO_ROOT / "data" / "other.db"


def test_repr_never_contains_secrets() -> None:
    text = repr(config.get_settings())
    for secret in ("test-anthropic-key", "test-openai-key", "test-tmdb-token"):
        assert secret not in text


def test_every_priced_model_has_input_and_output_price() -> None:
    for model in (config.AGENT_MODEL, config.EXPLAIN_MODEL, config.EMBED_MODEL):
        assert model in config.PRICES_PER_MTOK
    assert config.EMBED_DIMS == 1536
    assert abs(sum(config.SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def test_popularity_is_only_a_tie_breaker() -> None:
    """At 0.05 popularity reordered close vibe matches and cut vibe recall@10 from 0.53 to 0.41.

    It must stay positive (the 'Popular pick' fallback needs an order) but far below vibe.
    """
    weights = config.SCORE_WEIGHTS
    assert 0 < weights["popularity"] <= 0.005
    assert weights["popularity"] * 20 < weights["vibe"]
