"""The Chat page: the vibe prompt, the conversation, recommendation cards and thumbs feedback."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest
from tenacity import wait_none

from median import agent, feedback, llm, recommend
from median.config import get_settings
from median.db import connect, ensure_db
from tests.fakes import FakeStreamingAnthropic, agent_message, text_block, tool_use_block
from tests.fixtures.catalog import cluster_centroid, load_mini_catalog, read_catalog

REPO = Path(__file__).resolve().parent.parent
HORROR = 1


def open_chat(persona: int | None = None) -> AppTest:
    at = AppTest.from_file(str(REPO / "app" / "Home.py"), default_timeout=30)
    at.run()
    at.switch_page("pages/4_Chat.py")
    at.run()
    if persona is not None:
        at.sidebar.selectbox[0].select(persona).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def shown(at: AppTest) -> list[str]:
    return (
        [m.value for m in at.markdown] + [w.value for w in at.warning] + [i.value for i in at.info]
    )


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    ensure_db()
    c = connect(get_settings().db_path)
    load_mini_catalog(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_WAIT", wait_none())
    centroid = cluster_centroid(read_catalog()["clusters"]["haunted"])
    monkeypatch.setattr(recommend, "embed_query", lambda conn, text: centroid)


def script(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> FakeStreamingAnthropic:
    client = FakeStreamingAnthropic(list(replies))
    monkeypatch.setattr(llm, "_anthropic_client", lambda: client)
    return client


def picks(*ids: int) -> list[Any]:
    """A model that searches by mood for both media, then presents `ids`."""
    mood = {"mood_query": "Mood: eerie, tense", "limit": 6}
    return [
        agent_message(
            tool_use_block("recommend_for_persona", {"target_media": "movie", **mood}, "t_movie"),
            tool_use_block("recommend_for_persona", {"target_media": "book", **mood}, "t_book"),
        ),
        agent_message(
            text_block("Eerie and tense: one film, one book."),
            tool_use_block(
                "present_recommendations",
                {"work_ids": list(ids), "blurbs": {str(i): f"blurb {i}" for i in ids}},
                "t_show",
            ),
        ),
    ]


def pick_vibes(at: AppTest, *vibes: str) -> AppTest:
    return at.button_group(key=f"vibe-pills-{HORROR}").set_value(list(vibes)).run()


def find_picks(at: AppTest) -> AppTest:
    return at.button[0].click().run()


# Opening ----------------------------------------------------------------------------------------


def test_without_embeddings_the_page_explains_what_to_run() -> None:
    at = open_chat()
    assert any("no vibe embeddings" in v for v in shown(at))
    assert not at.chat_input


def test_it_greets_the_user_and_asks_for_vibes(conn) -> None:
    at = open_chat()
    assert agent.GREETING in shown(at)
    pills = at.button_group(key=f"vibe-pills-{HORROR}")
    assert "eerie" in pills.options and len(pills.options) <= 24  # real vibes from the catalog
    assert at.button_group(key=f"vibe-media-{HORROR}").value == "Movies and books"
    assert at.button[0].label == "Find my picks" and at.button[0].disabled  # needs a vibe first
    assert len(at.chat_input) == 1


def test_picking_vibes_enables_the_button(conn) -> None:
    at = pick_vibes(open_chat(), "eerie", "tense")
    assert not at.button[0].disabled


# Recommending -----------------------------------------------------------------------------------


def test_picked_vibes_produce_cards_for_both_media(conn, monkeypatch) -> None:
    client = script(monkeypatch, *picks(1, 11))
    at = pick_vibes(open_chat(), "eerie", "tense")
    at.button_group(key=f"vibe-media-{HORROR}").set_value("Movies and books")
    at = find_picks(at)
    assert not at.exception, [e.value for e in at.exception]

    sent = client.calls[0]["messages"][-1]
    assert sent == {
        "role": "user",
        "content": "I'm in the mood for: eerie, tense. Show me both movies and books.",
    }
    text = shown(at)
    assert sent["content"] in text
    assert "Eerie and tense: one film, one book." in text
    assert "blurb 1" in text and "blurb 11" in text
    assert len(at.page_link) == 2  # one linked title per card, loaded from the DB rows
    assert "Matches your mood request" in " ".join(c.value for c in at.caption)
    assert not at.button_group  # the picker goes away once the conversation starts


def test_a_medium_choice_is_passed_to_the_agent(conn, monkeypatch) -> None:
    client = script(monkeypatch, *picks(11))
    at = pick_vibes(open_chat(), "eerie")
    at.button_group(key=f"vibe-media-{HORROR}").set_value("Books")
    find_picks(at)
    assert client.calls[0]["messages"][-1]["content"].endswith("Only books, please.")


def test_a_typed_vague_request_gets_a_vibe_question_and_no_cards(conn, monkeypatch) -> None:
    question = "What vibe are you after: eerie, cozy or melancholic?"
    script(monkeypatch, agent_message(text_block(question)))
    at = open_chat()
    at.chat_input[0].set_value("recommend me something").run()
    assert question in shown(at)
    assert not at.page_link and not at.button_group


def test_a_follow_up_sends_the_history_including_which_cards_were_shown(conn, monkeypatch) -> None:
    client = script(monkeypatch, *picks(1, 11), agent_message(text_block("Darker it is.")))
    at = find_picks(pick_vibes(open_chat(), "eerie"))
    at.chat_input[0].set_value("darker").run()
    assert "Darker it is." in shown(at)
    follow_up = client.calls[-1]["messages"]
    assert [m["role"] for m in follow_up] == ["user", "assistant", "user"]
    assert (
        "[Cards shown to the user: The Shining (movie), The Shining (book)]"
        in follow_up[1]["content"]
    )
    assert len(at.page_link) == 2  # the earlier cards are still on the page


def test_both_sides_of_the_chat_are_logged(conn, monkeypatch) -> None:
    script(monkeypatch, *picks(1, 11))
    find_picks(pick_vibes(open_chat(), "eerie"))
    rows = conn.execute("SELECT persona_id, role, recommended_ids FROM chat_messages ORDER BY id")
    assert [(r["persona_id"], r["role"], r["recommended_ids"]) for r in rows] == [
        (HORROR, "user", "[]"),
        (HORROR, "assistant", "[1, 11]"),
    ]
    usage = conn.execute("SELECT DISTINCT purpose FROM llm_usage").fetchall()
    assert [r["purpose"] for r in usage] == ["chat_agent"]


def test_a_model_failure_is_shown_as_a_warning(conn, monkeypatch) -> None:
    monkeypatch.setattr(
        llm, "_anthropic_client", lambda: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    at = open_chat()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert any("couldn't reach the language model" in w.value for w in at.warning)


# Feedback and conversations -------------------------------------------------------------------


def test_thumbs_save_a_vote_and_pressing_again_clears_it(conn, monkeypatch) -> None:
    script(monkeypatch, *picks(1, 11))
    at = find_picks(pick_vibes(open_chat(), "eerie"))
    key = f"vote-up-chat-1-{HORROR}-1"  # the card's key is the message index (1)
    at.button(key=key).click().run()
    assert feedback.get_signal(conn, HORROR, 1) == "up"
    assert at.button(key=key).proto.type == "primary"
    at.button(key=f"vote-down-chat-1-{HORROR}-1").click().run()
    assert feedback.get_signal(conn, HORROR, 1) == "down"
    at.button(key=f"vote-down-chat-1-{HORROR}-1").click().run()
    assert feedback.get_signal(conn, HORROR, 1) is None
    assert not at.exception


def test_each_persona_has_its_own_conversation(conn, monkeypatch) -> None:
    script(monkeypatch, *picks(1, 11))
    at = find_picks(pick_vibes(open_chat(), "eerie"))
    assert at.page_link
    at.sidebar.selectbox[0].select(2).run()
    assert not at.page_link and at.button_group  # persona 2 starts fresh, with the vibe picker
    sessions = at.session_state["chat_sessions"]
    assert [m["role"] for m in sessions[1]["messages"]] == ["user", "assistant"]  # still kept
    assert sessions[2]["messages"] == [] and sessions[1]["id"] != sessions[2]["id"]


def test_start_over_clears_the_conversation(conn, monkeypatch) -> None:
    script(monkeypatch, *picks(1, 11))
    at = find_picks(pick_vibes(open_chat(), "eerie"))
    at.sidebar.button[0].click().run()
    assert not at.page_link and at.button_group
