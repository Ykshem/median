from __future__ import annotations

import sqlite3

import pytest

from median import config, llm
from tests.fakes import FakeBatchClient


def _requests(n: int = 3) -> list[llm.BatchRequest]:
    return [
        llm.BatchRequest(f"w{i}", [{"role": "user", "content": f"hi {i}"}], max_tokens=50)
        for i in range(n)
    ]


def _run(db: sqlite3.Connection, **kw):
    return llm.run_batch(
        db,
        purpose="vibe_card",
        tag="v1-try1",
        model=config.AGENT_MODEL,
        requests=kw.pop("requests", _requests()),
        sleep=lambda s: None,
        **kw,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    def install(answer=lambda r: f"reply to {r['custom_id']}", **kw) -> FakeBatchClient:
        fake = FakeBatchClient(answer, **kw)
        monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
        return fake

    return install


def test_run_batch_submits_polls_and_returns_results_by_custom_id(db, client) -> None:
    fake = client(statuses=("in_progress", "in_progress", "ended"))
    results = _run(db)
    assert [r.custom_id for r in results] == ["w0", "w1", "w2"]
    assert all(r.status == "succeeded" for r in results)
    assert results[1].text == "reply to w1"
    assert fake.retrieves == 3  # polled until it ended
    sent = fake.created[0][0]
    assert sent["custom_id"] == "w0"
    assert sent["params"] == {
        "model": config.AGENT_MODEL,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi 0"}],
    }


def test_system_prompt_is_forwarded(db, client) -> None:
    fake = client()
    request = llm.BatchRequest("w1", [{"role": "user", "content": "x"}], system="be brief")
    _run(db, requests=[request])
    assert fake.created[0][0]["params"]["system"] == "be brief"


def test_usage_is_logged_once_at_the_batch_price(db, client) -> None:
    client(lambda r: ("ok", 1_000_000, 1_000_000))
    _run(db, requests=_requests(1))
    (row,) = db.execute("SELECT * FROM llm_usage").fetchall()
    assert row["purpose"] == "vibe_card"
    assert (row["input_tokens"], row["output_tokens"]) == (1_000_000, 1_000_000)
    assert row["cost_usd"] == pytest.approx(0.30)  # half of the $0.60 regular price


def test_rerun_resumes_the_same_batch_and_does_not_pay_or_log_twice(db, client) -> None:
    fake = client()
    first = _run(db)
    second = _run(db)
    assert len(fake.created) == 1  # no second create
    assert [r.text for r in first] == [r.text for r in second]
    assert db.execute("SELECT COUNT(*) FROM llm_usage").fetchone()[0] == 1


def test_errored_requests_are_reported_and_not_billed(db, client) -> None:
    client(lambda r: None if r["custom_id"] == "w1" else "fine")
    results = _run(db)
    by_id = {r.custom_id: r for r in results}
    assert by_id["w1"].status == "errored" and by_id["w1"].text is None
    assert "boom" in by_id["w1"].error
    assert by_id["w0"].status == "succeeded"
    (row,) = db.execute("SELECT input_tokens FROM llm_usage").fetchall()
    assert row["input_tokens"] == 800  # two succeeded requests x 400


def test_budget_guard_blocks_submission(db, client) -> None:
    fake = client()
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('x', 'm', 5)")
    db.commit()
    with pytest.raises(llm.BudgetExceeded):
        _run(db, estimated_usd=0.1)
    assert fake.created == []
    assert db.execute("SELECT COUNT(*) FROM meta WHERE key LIKE 'batch:%'").fetchone()[0] == 0


def test_empty_and_oversized_batches_are_rejected(db, client) -> None:
    client()
    with pytest.raises(ValueError, match="empty"):
        _run(db, requests=[])
    too_many = [llm.BatchRequest(f"w{i}", []) for i in range(llm.MAX_BATCH_REQUESTS + 1)]
    with pytest.raises(ValueError, match="split"):
        _run(db, requests=too_many)


def test_wait_times_out(client) -> None:
    client(statuses=("in_progress",))
    now = {"t": 0.0}

    def clock() -> float:
        now["t"] += 100
        return now["t"]

    with pytest.raises(TimeoutError):
        llm.wait_for_batch("msgbatch_1", timeout_seconds=250, sleep=lambda s: None, clock=clock)
