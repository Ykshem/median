"""Agent guardrails: the model can only recommend what its tools returned this turn."""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import httpx
import pytest
from tenacity import wait_none

from median import agent, config, llm, recommend
from tests.fakes import FakeStreamingAnthropic, agent_message, text_block, tool_use_block
from tests.fixtures.catalog import cluster_centroid, read_catalog

PERSONA = 1


@pytest.fixture(autouse=True)
def _setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_WAIT", wait_none())
    centroid = cluster_centroid(read_catalog()["clusters"]["haunted"])
    monkeypatch.setattr(recommend, "embed_query", lambda conn, text: centroid)


def script(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> FakeStreamingAnthropic:
    client = FakeStreamingAnthropic(list(replies))
    monkeypatch.setattr(llm, "_anthropic_client", lambda: client)
    return client


def find(media: str = "any", tool_id: str = "toolu_find", limit: int = 6) -> Any:
    args = {"target_media": media, "mood_query": "eerie", "limit": limit}
    return tool_use_block("recommend_for_persona", args, tool_id)


def present(ids: list[Any], tool_id: str = "toolu_show") -> Any:
    return tool_use_block(
        "present_recommendations", {"work_ids": ids, "blurbs": {str(i): "x" for i in ids}}, tool_id
    )


def test_ids_no_tool_returned_are_dropped_and_logged(
    db, mini_catalog, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The two nearest movies are 1 and 2. 5 is a real work (Fellowship of the Ring) that no tool
    # returned, 9999 doesn't exist, and "2", True are not ids at all.
    script(
        monkeypatch,
        agent_message(find("movie", limit=2)),
        agent_message(text_block("Picks."), present([1, 5, 9999, "2", True, 2])),
    )
    with caplog.at_level(logging.WARNING, logger="median.agent"):
        result = agent.run_turn(db, PERSONA, [], "eerie")
    assert [c.work_id for c in result.cards] == [1, 2]
    assert result.dropped_ids == [5, 9999]
    assert "dropped [5, 9999]" in caplog.text


def test_ids_remembered_from_an_earlier_turn_must_come_from_a_tool_again(
    db, mini_catalog, monkeypatch
) -> None:
    history = [
        {"role": "user", "content": "eerie"},
        {
            "role": "assistant",
            "content": "Try The Shining.\n[Cards shown to the user: The Shining]",
        },
    ]
    client = script(
        monkeypatch,
        agent_message(present([1])),  # straight to the cards, with no tool call this turn
        agent_message(find("movie", "toolu_2")),
        agent_message(text_block("Here."), present([1], "toolu_3")),
    )
    result = agent.run_turn(db, PERSONA, history, "show me that again")
    # the first attempt was refused and the model was told why, then it searched and tried again
    refusal = client.calls[1]["messages"][-1]["content"][0]
    assert refusal["is_error"] is True and "tool results this turn" in refusal["content"]
    assert [c.work_id for c in result.cards] == [1]


def test_at_most_six_cards_are_shown(db, mini_catalog, monkeypatch) -> None:
    script(
        monkeypatch,
        agent_message(find("any")),
        agent_message(text_block("Many."), present(list(range(1, 21)))),
    )
    result = agent.run_turn(db, PERSONA, [], "eerie")
    assert len(result.cards) <= config.AGENT_MAX_PRESENTED
    assert all(c.work_id in range(1, 21) for c in result.cards)


def test_a_tool_error_is_fed_back_and_the_model_can_recover(db, mini_catalog, monkeypatch) -> None:
    bad = tool_use_block("get_work", {"work_id": 9999}, "toolu_bad")
    client = script(
        monkeypatch,
        agent_message(bad),
        agent_message(find("movie", "toolu_ok")),
        agent_message(text_block("Done."), present([1])),
    )
    result = agent.run_turn(db, PERSONA, [], "eerie")
    first = client.calls[1]["messages"][-1]["content"][0]
    assert first["is_error"] is True and "no work with id 9999" in first["content"]
    assert [c.work_id for c in result.cards] == [1]


def test_the_assistant_turn_is_replayed_to_the_model_with_its_tool_calls(
    db, mini_catalog, monkeypatch
) -> None:
    client = script(
        monkeypatch,
        agent_message(text_block("Let me look."), find("movie", "toolu_a")),
        agent_message(text_block("Done."), present([1])),
    )
    agent.run_turn(db, PERSONA, [], "eerie")
    assistant = client.calls[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "Let me look."}
    assert assistant["content"][1]["type"] == "tool_use"
    assert assistant["content"][1]["id"] == "toolu_a"
    assert client.calls[1]["messages"][2]["content"][0]["tool_use_id"] == "toolu_a"


def test_a_second_present_call_in_one_message_is_refused(db, mini_catalog, monkeypatch) -> None:
    client = script(
        monkeypatch,
        agent_message(find("movie")),
        agent_message(present([1], "toolu_1"), present([2], "toolu_2")),
    )
    result = agent.run_turn(db, PERSONA, [], "eerie")
    assert [c.work_id for c in result.cards] == [1]
    assert len(client.calls) == 2  # the first valid call ended the turn


def test_the_loop_stops_after_the_round_limit(db, mini_catalog, monkeypatch) -> None:
    client = script(monkeypatch, *[agent_message(find("movie", f"toolu_{i}")) for i in range(10)])
    result = agent.run_turn(db, PERSONA, [], "eerie")
    assert len(client.calls) == config.AGENT_MAX_ROUNDS
    assert result.error == "max_rounds" and result.cards == []
    assert result.text == agent.TANGLED_REPLY


def test_an_empty_reply_gets_a_friendly_message(db, mini_catalog, monkeypatch) -> None:
    script(monkeypatch, agent_message())
    result = agent.run_turn(db, PERSONA, [], "hello")
    assert result.text == agent.EMPTY_REPLY and result.cards == []


def test_the_daily_budget_stops_chat_before_any_call(db, mini_catalog, monkeypatch) -> None:
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('t', 'm', 5.0)")
    db.commit()
    client = script(monkeypatch, agent_message(text_block("never sent")))
    result = agent.run_turn(db, PERSONA, [], "eerie")
    assert result.error == "budget" and "spending limit" in result.text
    assert client.calls == []


def test_api_failures_are_retried_then_shown_as_a_friendly_error(
    db, mini_catalog, monkeypatch
) -> None:
    down = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    client = script(monkeypatch, down, down, down)
    result = agent.run_turn(db, PERSONA, [], "eerie")
    assert len(client.calls) == llm.MAX_ATTEMPTS
    assert result.error == "APIConnectionError" and "Browsing still works" in result.text
    assert result.cards == []


def test_a_transient_failure_before_any_text_is_retried(db, mini_catalog, monkeypatch) -> None:
    down = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    script(monkeypatch, down, agent_message(text_block("Back again.")))
    result = agent.run_turn(db, PERSONA, [], "hello")
    assert result.text == "Back again." and result.error is None


def test_a_missing_api_key_is_a_friendly_error(db, mini_catalog, monkeypatch) -> None:
    def no_key() -> None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(llm, "_anthropic_client", no_key)
    result = agent.run_turn(db, PERSONA, [], "hello")
    assert result.error == "RuntimeError" and ".env" in result.text


def test_only_the_last_twelve_history_messages_are_sent(db, mini_catalog, monkeypatch) -> None:
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(30)
    ]
    client = script(monkeypatch, agent_message(text_block("ok")))
    agent.run_turn(db, PERSONA, history, "now")
    sent = client.calls[0]["messages"]
    assert len(sent) == agent.MAX_HISTORY_MESSAGES + 1 and sent[0]["content"] == "m18"


def test_chat_messages_are_logged_with_the_shown_ids(db, mini_catalog) -> None:
    agent.log_message(db, PERSONA, "s1", "user", "eerie")
    agent.log_message(db, PERSONA, "s1", "assistant", "Picks.", [1, 11])
    rows = db.execute("SELECT role, content, recommended_ids FROM chat_messages ORDER BY id")
    assert [(r["role"], r["recommended_ids"]) for r in rows] == [
        ("user", "[]"),
        ("assistant", "[1, 11]"),
    ]
