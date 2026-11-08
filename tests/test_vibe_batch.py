from __future__ import annotations

import json
import sqlite3

import pytest

from median import config, llm
from median.models import VibeCard
from pipeline.enrich import vibe_cards as vc
from pipeline.sample_seed import upsert_work
from tests.fakes import FakeBatchClient
from tests.test_vibe_cards import GOOD_CARD, fetched, make_work


def _works(db: sqlite3.Connection, n: int):
    works = []
    for i in range(1, n + 1):
        work_id = upsert_work(db, fetched(f"Title {i}", ext=str(i)))
        works.append(make_work(id=work_id, title=f"Title {i}", creators=[]))
    db.commit()
    return works


def _card(**overrides: object) -> str:
    return json.dumps({**GOOD_CARD, **overrides})


@pytest.fixture
def batch(monkeypatch: pytest.MonkeyPatch):
    def install(answer) -> FakeBatchClient:
        fake = FakeBatchClient(answer)
        monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
        return fake

    return install


def _run(db: sqlite3.Connection, works) -> vc.BatchOutcome:
    return vc.generate_cards_batch(db, works, vc.load_template(), sleep=lambda s: None)


def test_every_good_card_is_stored_from_a_single_batch(db: sqlite3.Connection, batch) -> None:
    fake = batch(lambda r: _card())
    outcome = _run(db, _works(db, 3))
    assert outcome.stored == 3 and outcome.failed == []
    assert len(fake.created) == 1 and len(fake.created[0]) == 3
    rows = db.execute("SELECT model, prompt_version, card_text FROM vibe_cards").fetchall()
    assert len(rows) == 3
    assert all(r["model"] == config.AGENT_MODEL and r["prompt_version"] == "v1" for r in rows)
    assert rows[0]["card_text"] == VibeCard(**GOOD_CARD).to_text()


def test_invalid_cards_are_retried_once_in_a_second_batch_with_the_reason(
    db: sqlite3.Connection, batch
) -> None:
    def answer(request):
        prompt = request["params"]["messages"][0]["content"]
        if request["custom_id"] == "w2" and "rejected" not in prompt:
            return _card(logline="A film about a family.")  # breaks the no-medium rule
        return _card()

    fake = batch(answer)
    works = _works(db, 3)
    outcome = _run(db, works)
    assert outcome.stored == 3 and outcome.failed == []
    assert [len(b) for b in fake.created] == [3, 1]
    retry_prompt = fake.created[1][0]["params"]["messages"][0]["content"]
    assert "mentions the medium" in retry_prompt


def test_a_card_that_is_still_bad_after_the_retry_is_reported_not_stored(
    db: sqlite3.Connection, batch
) -> None:
    fake = batch(lambda r: "not json" if r["custom_id"] == "w2" else _card())
    works = _works(db, 3)
    outcome = _run(db, works)
    assert outcome.stored == 2
    ((work, problems),) = outcome.failed
    assert work.id == works[1].id and "invalid card JSON" in problems[0]
    assert len(fake.created) == 2  # exactly one retry round, never more
    assert db.execute("SELECT COUNT(*) FROM vibe_cards").fetchone()[0] == 2


def test_errored_requests_get_the_retry_round_too(db: sqlite3.Connection, batch) -> None:
    calls = {"w1": 0}

    def answer(request):
        if request["custom_id"] == "w1":
            calls["w1"] += 1
            return None if calls["w1"] == 1 else _card()
        return _card()

    batch(answer)
    outcome = _run(db, _works(db, 2))
    assert outcome.stored == 2 and outcome.failed == []


def test_existing_cards_are_skipped_and_a_rerun_resumes_without_a_new_batch(
    db: sqlite3.Connection, batch
) -> None:
    fake = batch(lambda r: _card())
    works = _works(db, 3)
    vc.store_card(db, works[0].id, VibeCard(**GOOD_CARD), config.AGENT_MODEL)
    _run(db, works)
    assert len(fake.created[0]) == 2  # the finished work was not re-sent
    again = _run(db, works)
    assert again.stored == 0 and len(fake.created) == 1  # nothing left to generate


def test_large_sets_are_split_into_batches(
    db: sqlite3.Connection, batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "MAX_BATCH_REQUESTS", 2)
    fake = batch(lambda r: _card())
    outcome = _run(db, _works(db, 5))
    assert outcome.stored == 5
    assert [len(b) for b in fake.created] == [2, 2, 1]


def test_usage_is_logged_at_the_batch_price(db: sqlite3.Connection, batch) -> None:
    batch(lambda r: (_card(), 1_000_000, 1_000_000))
    _run(db, _works(db, 1))
    (row,) = db.execute("SELECT * FROM llm_usage WHERE purpose = 'vibe_card'").fetchall()
    assert row["cost_usd"] == pytest.approx(0.30)


def test_the_batch_estimate_is_half_the_regular_price() -> None:
    prompts = ["x" * 4000] * 10
    regular = vc.estimate(prompts)
    discounted = vc.estimate(prompts, batch=True)
    assert discounted.usd == pytest.approx(regular.usd / 2)
    short = vc.estimate(prompts, batch=True, output_tokens_each=60)
    assert short.output_tokens == 600


def test_the_budget_guard_stops_a_batch_before_it_is_submitted(
    db: sqlite3.Connection, batch
) -> None:
    fake = batch(lambda r: _card())
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('x', 'm', 5)")
    db.commit()
    with pytest.raises(llm.BudgetExceeded):
        _run(db, _works(db, 2))
    assert fake.created == []
