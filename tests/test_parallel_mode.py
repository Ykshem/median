"""The regular-API (parallel) path: run_parallel, and card / link generation through it."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import anthropic
import httpx
import pytest
from tenacity import wait_none

from median import config, llm
from median.models import VibeCard
from pipeline.enrich import link_check
from pipeline.enrich import vibe_cards as vc
from tests.fakes import FakeConcurrentClient
from tests.test_match_and_link_check import World, _three_links, _verdict
from tests.test_vibe_batch import _card, _works


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_WAIT", wait_none())


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    def install(answer, **kw) -> FakeConcurrentClient:
        fake = FakeConcurrentClient(answer, **kw)
        monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
        return fake

    return install


def requests(n: int = 4, **extra) -> list[llm.BatchRequest]:
    return [
        llm.BatchRequest(f"r{i}", [{"role": "user", "content": f"job {i}"}], max_tokens=77, **extra)
        for i in range(n)
    ]


def run(db: sqlite3.Connection, reqs: list[llm.BatchRequest], **kw):
    return llm.run_parallel(db, purpose="vibe_card", model=config.AGENT_MODEL, requests=reqs, **kw)


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://x.invalid"))


# run_parallel ------------------------------------------------------------------------------------


def test_every_request_gets_a_result_keyed_by_custom_id(db, client) -> None:
    fake = client(lambda kw: f"re: {kw['messages'][0]['content']}")
    results = run(db, requests(6), workers=3)
    assert {r.custom_id: r.text for r in results} == {f"r{i}": f"re: job {i}" for i in range(6)}
    assert all(r.status == "succeeded" for r in results) and len(fake.calls) == 6


def test_request_options_are_forwarded(db, client) -> None:
    fake = client(lambda kw: "ok")
    reqs = requests(
        1, system="be brief", thinking=llm.THINKING_OFF, output_config={"effort": "low"}
    )
    run(db, reqs)
    call = fake.calls[0]
    assert call["model"] == config.AGENT_MODEL and call["max_tokens"] == 77
    assert (call["system"], call["thinking"], call["output_config"]) == (
        "be brief",
        {"type": "disabled"},
        {"effort": "low"},
    )


def test_usage_is_logged_per_call_at_the_regular_price_not_the_batch_price(db, client) -> None:
    client(lambda kw: ("ok", 1_000_000, 1_000_000))
    run(db, requests(3))
    rows = db.execute("SELECT cost_usd FROM llm_usage WHERE purpose = 'vibe_card'").fetchall()
    assert len(rows) == 3
    assert all(r["cost_usd"] == pytest.approx(0.60) for r in rows)  # batch would be 0.30


def test_results_reach_on_result_on_the_calling_thread(db, client) -> None:
    client(lambda kw: "ok", delay=0.02)
    seen: list[tuple[str, int]] = []
    run(
        db,
        requests(6),
        workers=4,
        on_result=lambda r: seen.append((r.custom_id, threading.get_ident())),
    )
    assert len(seen) == 6
    assert {thread for _, thread in seen} == {threading.get_ident()}  # sqlite stays single-threaded


def test_requests_really_run_concurrently(db, client) -> None:
    fake = client(lambda kw: "ok", delay=0.05)
    run(db, requests(8), workers=4)
    assert len(fake.threads) > 1


def test_nothing_is_sent_when_the_budget_is_already_spent(db, client) -> None:
    fake = client(lambda kw: "ok")
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('x', 'm', 5)")
    db.commit()
    with pytest.raises(llm.BudgetExceeded):
        run(db, requests(3))
    assert fake.calls == []


def test_the_budget_trips_mid_run_without_losing_requests_already_sent(
    db, client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DAILY_USD_LIMIT", "0.0001")
    config.get_settings.cache_clear()
    fake = client(lambda kw: ("ok", 1_000_000, 0), delay=0.02)  # $0.10 each
    handled: list[str] = []
    with pytest.raises(llm.BudgetExceeded):
        run(db, requests(12), workers=2, on_result=lambda r: handled.append(r.custom_id))
    assert 1 <= len(fake.calls) < 12  # it stopped sending
    logged = db.execute("SELECT COUNT(*) FROM llm_usage").fetchone()[0]
    assert logged == len(fake.calls) == len(handled)  # every sent request was logged and handled


def test_transient_errors_are_retried(db, client) -> None:
    attempts: dict[str, int] = {}

    def answer(kw):
        key = kw["messages"][0]["content"]
        attempts[key] = attempts.get(key, 0) + 1
        return connection_error() if attempts[key] < 3 else "finally"

    fake = client(answer)
    results = run(db, requests(2))
    assert [r.text for r in results] == ["finally", "finally"]
    assert len(fake.calls) == 6  # three attempts each
    assert db.execute("SELECT COUNT(*) FROM llm_usage").fetchone()[0] == 2  # only successes billed


def test_a_request_that_keeps_failing_comes_back_errored_not_raised(
    db, client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "PARALLEL_ATTEMPTS", 2)
    client(lambda kw: connection_error() if "job 1" in kw["messages"][0]["content"] else "ok")
    by_id = {r.custom_id: r for r in run(db, requests(3))}
    assert by_id["r1"].status == "errored" and "APIConnectionError" in by_id["r1"].error
    assert by_id["r0"].status == by_id["r2"].status == "succeeded"
    assert db.execute("SELECT COUNT(*) FROM llm_usage").fetchone()[0] == 2


def test_a_fatal_error_such_as_no_credit_stops_the_run(db, client) -> None:
    fatal = anthropic.BadRequestError(
        "Your credit balance is too low",
        response=httpx.Response(400, request=httpx.Request("POST", "https://x.invalid")),
        body=None,
    )
    client(lambda kw: fatal if "job 0" in kw["messages"][0]["content"] else "ok")
    with pytest.raises(anthropic.BadRequestError, match="credit balance"):
        run(db, requests(3), workers=3)


def test_an_empty_request_list_does_nothing(db, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_anthropic_client", lambda: pytest.fail("no client needed"))
    assert run(db, []) == []


def test_run_parallel_can_stand_in_for_run_batch_arguments(db, client) -> None:
    client(lambda kw: "ok")
    results = run(db, requests(1), tag="ignored", poll_seconds=1.0, sleep=lambda s: None)
    assert len(results) == 1


# vibe cards through the regular API --------------------------------------------------------------


def card_answer(kw) -> str:
    return _card()


def generate(db, works, **kw):
    return vc.generate_cards_batch(
        db, works, vc.load_template(), dispatch=llm.run_parallel, sleep=lambda s: None, **kw
    )


def test_cards_are_generated_and_stored_through_the_regular_api(db, client) -> None:
    fake = client(card_answer)
    outcome = generate(db, _works(db, 5))
    assert outcome.stored == 5 and outcome.failed == []
    assert db.execute("SELECT COUNT(*) FROM vibe_cards").fetchone()[0] == 5
    assert all(c["thinking"] == {"type": "disabled"} for c in fake.calls)


def test_cards_are_billed_at_the_regular_price_when_not_batched(db, client) -> None:
    client(lambda kw: (_card(), 1_000_000, 1_000_000))
    generate(db, _works(db, 1))
    (row,) = db.execute("SELECT cost_usd FROM llm_usage WHERE purpose = 'vibe_card'").fetchall()
    assert row["cost_usd"] == pytest.approx(0.60)


def test_an_invalid_card_is_retried_once_with_the_reason(db, client) -> None:
    def answer(kw):
        prompt = kw["messages"][0]["content"]
        if "Title 2" in prompt and "rejected" not in prompt:
            return _card(logline="A film about a family.")
        return _card()

    fake = client(answer)
    outcome = generate(db, _works(db, 3))
    assert outcome.stored == 3 and outcome.failed == []
    assert len(fake.calls) == 4  # three first tries + one retry
    assert "mentions the medium" in fake.calls[-1]["messages"][0]["content"]


def test_a_card_still_bad_after_the_retry_is_reported_not_stored(db, client) -> None:
    fake = client(lambda kw: "not json" if "Title 2" in kw["messages"][0]["content"] else _card())
    outcome = generate(db, _works(db, 3))
    assert outcome.stored == 2 and len(outcome.failed) == 1
    assert len(fake.calls) == 4  # never a second retry


def test_a_fatal_error_keeps_the_cards_stored_before_it(db, client) -> None:
    fatal = anthropic.AuthenticationError(
        "invalid x-api-key",
        response=httpx.Response(401, request=httpx.Request("POST", "https://x.invalid")),
        body=None,
    )
    client(lambda kw: fatal if "Title 4" in kw["messages"][0]["content"] else _card())
    with pytest.raises(anthropic.AuthenticationError):
        generate(db, _works(db, 6))
    stored = db.execute("SELECT COUNT(*) FROM vibe_cards").fetchone()[0]
    assert stored == 5  # every other card was saved as it arrived; a rerun only needs the last one


def test_a_rerun_after_an_interruption_only_generates_the_missing_cards(db, client) -> None:
    works = _works(db, 4)
    vc.store_card(db, works[0].id, VibeCard(**json.loads(_card())), config.AGENT_MODEL)
    fake = client(card_answer)
    generate(db, works)
    assert len(fake.calls) == 3


# link check through the regular API --------------------------------------------------------------


def test_links_are_verified_through_the_regular_api(db, client, tmp_path: Path) -> None:
    world = World(db, tmp_path)
    _three_links(world)
    answers = {
        "Alpha": _verdict(True, 0.95),
        "Beta": _verdict(False, 0.9),
        "Gamma": _verdict(True, 0.4),
    }
    fake = client(
        lambda kw: next(
            v for k, v in answers.items() if f"film: {k} " in kw["messages"][0]["content"]
        )
    )
    report = link_check.check_links(
        db, world.client, cache_path=tmp_path / "v.json", dispatch=llm.run_parallel
    )
    assert (report.kept, report.dropped) == (1, 2) and len(fake.calls) == 3
    assert all(c["thinking"] == {"type": "disabled"} for c in fake.calls)
    world.client.close()


def test_the_regular_estimate_is_double_the_batch_estimate(db, tmp_path: Path) -> None:
    world = World(db, tmp_path)
    _three_links(world)
    _, batch_cost = link_check.estimate_link_check(db, world.client, tmp_path / "v.json")
    _, regular_cost = link_check.estimate_link_check(
        db, world.client, tmp_path / "v.json", batch=False
    )
    assert regular_cost == pytest.approx(batch_cost * 2)
    world.client.close()
