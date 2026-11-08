from __future__ import annotations

import sqlite3

import anthropic
import httpx
import pytest
from tenacity import wait_none

from median import config, llm
from tests.fakes import FakeAnthropic, FakeOpenAI, anthropic_response, embedding_response


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_WAIT", wait_none())


def _transient() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


def _usage_rows(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return db.execute("SELECT * FROM llm_usage ORDER BY id").fetchall()


# Cost math ---------------------------------------------------------------------------------------


def test_cost_math_for_each_model() -> None:
    assert llm.cost_usd(config.AGENT_MODEL, 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert llm.cost_usd(config.EXPLAIN_MODEL, 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert llm.cost_usd(config.EMBED_MODEL, 1_000_000, 0) == pytest.approx(0.02)


def test_batch_costs_half() -> None:
    full = llm.cost_usd(config.AGENT_MODEL, 1_000_000, 1_000_000)
    assert llm.cost_usd(config.AGENT_MODEL, 1_000_000, 1_000_000, batch=True) == pytest.approx(
        full / 2
    )


def test_unknown_model_raises_instead_of_costing_zero() -> None:
    with pytest.raises(ValueError, match="No price configured"):
        llm.cost_usd("claude-made-up", 10, 10)


# Budget guard ------------------------------------------------------------------------------------


def test_budget_guard_blocks_over_limit_and_allows_under(db: sqlite3.Connection) -> None:
    llm.check_budget(db)  # nothing spent
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('t', 'm', 1.9)")
    db.commit()
    llm.check_budget(db, estimated=0.05)  # 1.95 <= 2
    with pytest.raises(llm.BudgetExceeded, match="DAILY_USD_LIMIT"):
        llm.check_budget(db, estimated=0.2)  # 2.10 > 2


def test_budget_ignores_previous_days(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT INTO llm_usage (purpose, model, cost_usd, created_at) "
        "VALUES ('old', 'm', 50, datetime('now', '-2 days'))"
    )
    db.commit()
    assert llm.spent_today(db) == 0
    llm.check_budget(db)


def test_complete_refuses_without_calling_api_when_over_limit(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('t', 'm', 5)")
    db.commit()
    fake = FakeAnthropic([anthropic_response()])
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
    with pytest.raises(llm.BudgetExceeded):
        llm.complete(
            db, purpose="t", model=config.AGENT_MODEL, messages=[{"role": "user", "content": "hi"}]
        )
    assert fake.calls == []


# complete() --------------------------------------------------------------------------------------


def test_complete_logs_usage_and_forwards_arguments(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAnthropic([anthropic_response("hello", input_tokens=1000, output_tokens=200)])
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
    response = llm.complete(
        db,
        purpose="agent",
        model=config.AGENT_MODEL,
        system="be brief",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
    )
    assert response.content[0].text == "hello"
    assert fake.calls[0]["system"] == "be brief"
    assert fake.calls[0]["max_tokens"] == 50
    assert "tools" not in fake.calls[0]
    (row,) = _usage_rows(db)
    assert (row["purpose"], row["model"]) == ("agent", config.AGENT_MODEL)
    assert (row["input_tokens"], row["output_tokens"]) == (1000, 200)
    assert row["cost_usd"] == pytest.approx(llm.cost_usd(config.AGENT_MODEL, 1000, 200))


def test_complete_retries_transient_errors_then_succeeds(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAnthropic([_transient(), _transient(), anthropic_response()])
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
    llm.complete(
        db, purpose="t", model=config.AGENT_MODEL, messages=[{"role": "user", "content": "x"}]
    )
    assert len(fake.calls) == 3
    assert len(_usage_rows(db)) == 1  # failed attempts are not billed or logged


def test_complete_gives_up_after_three_attempts(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAnthropic([_transient(), _transient(), _transient(), anthropic_response()])
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
    with pytest.raises(anthropic.APIConnectionError):
        llm.complete(
            db, purpose="t", model=config.AGENT_MODEL, messages=[{"role": "user", "content": "x"}]
        )
    assert len(fake.calls) == 3
    assert _usage_rows(db) == []


def test_missing_api_key_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    config.get_settings.cache_clear()
    llm._anthropic_client.cache_clear()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm._anthropic_client()
    llm._anthropic_client.cache_clear()


# embed() -----------------------------------------------------------------------------------------


def test_embed_batches_at_256_and_logs_each_request(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOpenAI([embedding_response(256, prompt_tokens=1000), embedding_response(4)])
    monkeypatch.setattr(llm, "_openai_client", lambda: fake)
    vectors = llm.embed(db, [f"text {i}" for i in range(260)], purpose="embed")
    assert len(vectors) == 260
    assert [len(c["input"]) for c in fake.calls] == [256, 4]
    assert fake.calls[0]["model"] == config.EMBED_MODEL
    rows = _usage_rows(db)
    assert len(rows) == 2
    assert rows[0]["cost_usd"] == pytest.approx(1000 * 0.02 / 1_000_000)


def test_embed_rejects_wrong_dimensions(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOpenAI([embedding_response(1, dims=8)])
    monkeypatch.setattr(llm, "_openai_client", lambda: fake)
    with pytest.raises(ValueError, match="1536"):
        llm.embed(db, ["x"], purpose="embed")


def test_embed_empty_input_makes_no_calls(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOpenAI([])
    monkeypatch.setattr(llm, "_openai_client", lambda: fake)
    assert llm.embed(db, [], purpose="embed") == []
    assert fake.calls == []
