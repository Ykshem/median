"""Chat with Median: say what vibe you want, get movies and books that feel like it."""

from __future__ import annotations

import uuid

import streamlit as st

from app.components import db_conn, page_setup, render_recommendation_card
from median import agent, browse

persona_id = page_setup("Chat", "💬")
if persona_id is None:
    st.stop()

TOOL_LABELS = {
    "search_works": "Looking up titles…",
    "get_work": "Reading a vibe card…",
    "get_overlap": "Finding connections…",
    "recommend_for_persona": "Matching your vibe…",
    "get_persona_shelf": "Checking your shelves…",
}
MEDIA_CHOICES = {"Movies and books": "any", "Movies": "movie", "Books": "book"}

with db_conn() as conn:
    works = browse.count_vectors(conn)
    menu = agent.vibe_menu(conn)

if works == 0:
    st.info(
        "The catalog has no vibe embeddings yet, so vibe search can't work. "
        "Run `uv run python -m pipeline.build --sample` first."
    )
    st.stop()

# One conversation per persona, kept for the browser session.
sessions = st.session_state.setdefault("chat_sessions", {})
session = sessions.setdefault(persona_id, {"id": uuid.uuid4().hex, "messages": []})
messages: list[dict] = session["messages"]

if st.sidebar.button("Start over", disabled=not messages):
    sessions.pop(persona_id, None)
    st.rerun()


def render_cards(cards: list[agent.Card], index: int) -> None:
    """Draw recommendation cards from DB rows (never from model-written titles)."""
    with db_conn() as conn:
        by_id = browse.works_by_ids(conn, [c.work_id for c in cards])
    for card in cards:
        if card.work_id in by_id:
            render_recommendation_card(
                by_id[card.work_id],
                key=f"chat-{index}",
                persona_id=persona_id,
                blurb=card.blurb,
                reasons=card.reasons,
            )
    if cards:
        st.caption("👍 and 👎 teach Median your taste, so the next picks fit better.")


with st.chat_message("assistant"):
    st.markdown(agent.GREETING)

for i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])
        render_cards(message.get("cards", []), i)

# The vibe picker is offered until the conversation starts.
pending: str | None = None
picker = st.empty()
if not messages:
    with picker.container():
        picked = st.pills(
            "Pick the vibes you're in the mood for",
            menu.all(),
            selection_mode="multi",
            key=f"vibe-pills-{persona_id}",
        )
        media_label = st.segmented_control(
            "Show me",
            list(MEDIA_CHOICES),
            default="Movies and books",
            key=f"vibe-media-{persona_id}",
        )
        if st.button("Find my picks", type="primary", disabled=not picked):
            pending = agent.vibe_request(picked, MEDIA_CHOICES[media_label or "Movies and books"])

typed = st.chat_input("Describe a vibe, or name a movie or book you loved")
pending = pending or typed

if pending:
    picker.empty()
    history = [{"role": m["role"], "content": m["history"]} for m in messages]
    messages.append({"role": "user", "content": pending, "history": pending})
    with st.chat_message("user"):
        st.markdown(pending)
    with st.chat_message("assistant"):
        status = st.status("Thinking…")
        box = st.empty()
        streamed: list[str] = []

        def on_text(delta: str) -> None:
            streamed.append(delta)
            box.markdown("".join(streamed) + "▌")

        def on_tool(name: str) -> None:
            streamed.clear()  # text before a tool call is a preamble, not the answer
            box.empty()
            status.update(label=TOOL_LABELS.get(name, "Working…"))

        with db_conn() as conn:
            result = agent.run_turn(
                conn, persona_id, history, pending, on_text=on_text, on_tool=on_tool
            )
            agent.log_message(conn, persona_id, session["id"], "user", pending)
            agent.log_message(
                conn,
                persona_id,
                session["id"],
                "assistant",
                result.text,
                [c.work_id for c in result.cards],
            )
            history_text = agent.history_entry(conn, result)
    messages.append(
        {
            "role": "assistant",
            "content": result.text,
            "history": history_text,
            "cards": result.cards,
            "error": result.error is not None,
        }
    )
    st.rerun()  # redraw from the saved conversation: cards, the sidebar button and no picker
