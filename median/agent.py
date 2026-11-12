"""Chat agent: asks what vibe the user wants, then recommends movies and books for it.

Haiku runs a tool loop over the catalog (ARCHITECTURE.md section 6). The model can only *recommend*
works its tools returned in the current turn: `present_recommendations` is validated here, any other
id is dropped and logged, and the UI draws cards from DB rows, never from model-written titles.
Catalog text reaches the model only inside <work_data> tags. Reads the catalog; the only writes are
chat messages (`log_message`) and LLM usage (via `median.llm`).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic

from median import browse, llm, recommend, shelves, taste
from median.config import (
    AGENT_MAX_PRESENTED,
    AGENT_MAX_ROUNDS,
    AGENT_MAX_TOKENS,
    AGENT_MODEL,
    AGENT_MOOD_WEIGHT,
    AGENT_TOOL_RESULT_CHARS,
    AGENT_VIBE_MENU,
    PROMPTS_DIR,
)

logger = logging.getLogger(__name__)

PRESENT = "present_recommendations"
MAX_TOOL_LIMIT = 10
MAX_HISTORY_MESSAGES = 12
MEDIA = ("movie", "book", "any")

GREETING = (
    "Hi! I'm Median. Tell me what **vibe** you're in the mood for and I'll find movies and books "
    'that feel like it. Pick a few below, describe it in your own words ("cozy and a little '
    'melancholy, set in a small town"), or name a film or book you loved.'
)
EMPTY_REPLY = "Sorry, I didn't get a reply that time. Could you say that again?"
TANGLED_REPLY = (
    "I got a bit tangled looking that up. Could you describe the vibe again in a few words?"
)
NO_PICKS_REPLY = "I couldn't find picks I could verify for that. What vibe should I aim for?"


# Tool schemas -----------------------------------------------------------------------------------

_MEDIA_PROP = {"type": "string", "enum": list(MEDIA)}
_LIMIT_PROP = {"type": "integer", "minimum": 1, "maximum": MAX_TOOL_LIMIT}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_works",
        "description": (
            "Find works by title or creator, then by similar vibe. Use it to look up a movie or "
            "book the user names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "media_type": {**_MEDIA_PROP, "description": "Default any."},
                "limit": _LIMIT_PROP,
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_work",
        "description": "Get one work's details: vibe card (themes, mood, tone...) and its links.",
        "input_schema": {
            "type": "object",
            "properties": {"work_id": {"type": "integer"}},
            "required": ["work_id"],
        },
    },
    {
        "name": "get_overlap",
        "description": (
            "Find works that connect to a work the user liked: adaptations, same series, and "
            "similar vibe. Set target_media to the other medium to cross over."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "work_id": {"type": "integer"},
                "target_media": {**_MEDIA_PROP, "description": "Default any."},
                "limit": _LIMIT_PROP,
            },
            "required": ["work_id"],
        },
    },
    {
        "name": "recommend_for_persona",
        "description": (
            "Recommend works for the current persona, steered by the vibe the user asked for. "
            "Write mood_query like a vibe card, e.g. 'Themes: grief. Mood: eerie, melancholic.'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_media": {**_MEDIA_PROP, "description": "Default any."},
                "mood_query": {"type": "string"},
                "limit": _LIMIT_PROP,
            },
        },
    },
    {
        "name": "get_persona_shelf",
        "description": "List the works on the persona's shelves, with ratings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "shelf": {"type": "string", "enum": list(shelves.SHELVES)},
            },
        },
    },
    {
        "name": PRESENT,
        "description": (
            f"FINAL step: show up to {AGENT_MAX_PRESENTED} works to the user as cards. Only ids "
            "returned by the other tools this turn are accepted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "work_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": AGENT_MAX_PRESENTED,
                },
                "blurbs": {
                    "type": "object",
                    "description": "work id (as a string) -> one spoiler-free sentence",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["work_ids", "blurbs"],
        },
    },
]


# Results ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Card:
    """One recommendation to show: the work id (the UI loads the row), a blurb and the reasons."""

    work_id: int
    blurb: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TurnResult:
    """What one user turn produced. `error` is set when the reply is a friendly failure message."""

    text: str
    cards: list[Card] = field(default_factory=list)
    dropped_ids: list[int] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None


class ToolError(ValueError):
    """A tool call the model got wrong (bad arguments, unknown id). Reported back, not raised."""


# Vibe options -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class VibeMenu:
    """The mood and theme words offered to the user as vibe options."""

    moods: list[str]
    themes: list[str]

    def all(self) -> list[str]:
        """Return moods then themes as one list, without repeats."""
        return list(dict.fromkeys([*self.moods, *self.themes]))


def vibe_menu(conn: sqlite3.Connection, size: int = AGENT_VIBE_MENU) -> VibeMenu:
    """Return the most common moods and themes in the catalog (empty lists for an empty catalog)."""
    half = max(1, size // 2)
    moods = browse.list_moods(conn, half)
    themes = [t for t in browse.list_themes(conn, half + len(moods)) if t not in moods][:half]
    return VibeMenu(moods, themes)


def vibe_request(vibes: list[str], media: str = "any") -> str:
    """Return the chat message sent when the user picks vibe chips (and optionally one medium)."""
    if media not in MEDIA:
        raise ValueError(f"media must be one of {MEDIA}")
    medium = {
        "any": "Show me both movies and books.",
        "movie": "Only movies, please.",
        "book": "Only books, please.",
    }[media]
    return f"I'm in the mood for: {', '.join(vibes)}. {medium}"


# System prompt ----------------------------------------------------------------------------------


def _data(text: str) -> str:
    """Wrap catalog-derived text in <work_data> tags, with '<' and '>' neutralised."""
    return "<work_data>" + text.replace("<", "(").replace(">", ")") + "</work_data>"


def taste_summary(conn: sqlite3.Connection, persona_id: int) -> str:
    """Return the persona's taste in words (top themes and moods), or say there isn't one yet."""
    profile = taste.profile(conn, persona_id)
    if not profile.has_vector:
        return "No taste profile yet (fewer than 3 ratings), so ask rather than assume."
    themes = ", ".join(t for t, _ in profile.themes[:6])
    moods = ", ".join(m for m, _ in profile.moods[:6])
    return f"Tends to like themes: {themes}. Moods: {moods}."


def build_system_prompt(conn: sqlite3.Connection, persona_id: int) -> str:
    """Fill `prompts/agent_system.md` with the persona and the catalog's vibe words."""
    row = conn.execute(
        "SELECT name, description FROM personas WHERE id = ?", (persona_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown persona {persona_id}")
    menu = vibe_menu(conn)
    values = {
        "persona_name": row["name"],
        "persona_description": row["description"] or "no description",
        "taste_summary": _data(taste_summary(conn, persona_id)),
        "mood_menu": _data(", ".join(menu.moods) or "none yet"),
        "theme_menu": _data(", ".join(menu.themes) or "none yet"),
    }
    text = (PROMPTS_DIR / "agent_system.md").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


# Tools ------------------------------------------------------------------------------------------


def _text_arg(args: dict[str, Any], name: str, *, required: bool = False) -> str:
    value = args.get(name)
    if value is None or value == "":
        if required:
            raise ToolError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise ToolError(f"{name} must be a string")
    return value.strip()


def _int_arg(args: dict[str, Any], name: str) -> int:
    value = args.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{name} must be an integer")
    return value


def _media_arg(args: dict[str, Any], name: str) -> str:
    value = args.get(name) or "any"
    if value not in MEDIA:
        raise ToolError(f"{name} must be one of {list(MEDIA)}")
    return value


def _limit_arg(args: dict[str, Any]) -> int:
    value = args.get("limit", 6)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError("limit must be an integer")
    return max(1, min(MAX_TOOL_LIMIT, value))


def _summary(
    conn: sqlite3.Connection, work_id: int, reasons: list[str] | None = None
) -> dict[str, Any] | None:
    """Return a compact description of one work for the model, or None if it doesn't exist."""
    work = browse.get_work(conn, work_id)
    if work is None:
        return None
    card = browse.get_card(conn, work_id)
    item: dict[str, Any] = {
        "id": work.id,
        "title": work.title,
        "type": work.media_type,
        "year": work.year,
        "creators": work.creators[:2],
    }
    if card is not None:
        item["themes"] = card.themes
        item["mood"] = card.mood
        item["tone"] = card.tone
    if reasons:
        item["reasons"] = reasons
    return item


def _summaries(
    conn: sqlite3.Connection, ids: list[int], reasons: dict[int, list[str]] | None = None
) -> list[dict[str, Any]]:
    items = [_summary(conn, i, (reasons or {}).get(i)) for i in ids]
    return [item for item in items if item is not None]


def _search_works(conn: sqlite3.Connection, persona_id: int, args: dict[str, Any]) -> list[dict]:
    query = _text_arg(args, "query", required=True)
    media = _media_arg(args, "media_type")
    limit = _limit_arg(args)
    ids: list[int] = []
    if browse.fts_query(query) is not None:
        found = browse.search_works(
            conn, browse.SearchFilters(query=query, media_type=media), limit
        )
        ids = [w.id for w in found]
    if len(ids) < limit and browse.count_vectors(conn):
        try:
            vector = recommend.embed_query(conn, query)
        except Exception:  # keep the title matches: semantic search is a bonus
            logger.warning("semantic search unavailable for this query", exc_info=True)
        else:
            for work_id in browse.semantic_search(conn, vector, media, limit):
                if work_id not in ids and len(ids) < limit:
                    ids.append(work_id)
    return _summaries(conn, ids)


def _get_work(conn: sqlite3.Connection, persona_id: int, args: dict[str, Any]) -> list[dict]:
    work_id = _int_arg(args, "work_id")
    item = _summary(conn, work_id)
    if item is None:
        raise ToolError(f"no work with id {work_id}")
    item["links"] = [
        {"id": lw.work.id, "title": lw.work.title, "type": lw.work.media_type, "how": lw.relation}
        for lw in browse.linked_works(conn, work_id)
    ]
    return [item]


def _get_overlap(conn: sqlite3.Connection, persona_id: int, args: dict[str, Any]) -> list[dict]:
    work_id = _int_arg(args, "work_id")
    try:
        recs = recommend.overlap(
            conn, work_id, _media_arg(args, "target_media"), persona_id, k=_limit_arg(args)
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return _summaries(conn, [r.work_id for r in recs], {r.work_id: r.reasons for r in recs})


def _recommend_for_persona(
    conn: sqlite3.Connection, persona_id: int, args: dict[str, Any]
) -> list[dict]:
    media = _media_arg(args, "target_media")
    mood = _text_arg(args, "mood_query")
    vector = None
    if mood:
        try:
            vector = recommend.embed_query(conn, mood)
        except Exception as exc:
            logger.warning("mood embedding failed", exc_info=True)
            raise ToolError("mood search is unavailable right now; try without mood_query") from exc
    recs = recommend.for_persona(
        conn,
        persona_id,
        media,
        mood_vector=vector,
        k=_limit_arg(args),
        mood_weight=AGENT_MOOD_WEIGHT,
    )
    return _summaries(conn, [r.work_id for r in recs], {r.work_id: r.reasons for r in recs})


def _get_persona_shelf(
    conn: sqlite3.Connection, persona_id: int, args: dict[str, Any]
) -> list[dict]:
    shelf = _text_arg(args, "shelf") or None
    if shelf is not None and shelf not in shelves.SHELVES:
        raise ToolError(f"shelf must be one of {list(shelves.SHELVES)}")
    items = []
    for entry in shelves.list_shelf(conn, persona_id, shelf)[:20]:
        item = {
            "id": entry.work.id,
            "title": entry.work.title,
            "type": entry.work.media_type,
            "year": entry.work.year,
            "shelf": entry.item.shelf,
        }
        if entry.item.rating is not None:
            item["rating"] = entry.item.rating
        items.append(item)
    return items


_HANDLERS: dict[str, Callable[[sqlite3.Connection, int, dict[str, Any]], list[dict]]] = {
    "search_works": _search_works,
    "get_work": _get_work,
    "get_overlap": _get_overlap,
    "recommend_for_persona": _recommend_for_persona,
    "get_persona_shelf": _get_persona_shelf,
}


def render_tool_result(items: list[dict[str, Any]], limit: int | None = None) -> str:
    """Return `items` as JSON inside <work_data> tags, dropping items from the end to fit `limit`.

    `limit` defaults to AGENT_TOOL_RESULT_CHARS. '<' and '>' are escaped as \\u003c and \\u003e, so
    catalog text can't close the tag early.
    """
    limit = AGENT_TOOL_RESULT_CHARS if limit is None else limit
    kept = list(items)
    while True:
        body = json.dumps({"results": kept}, ensure_ascii=False)
        text = "<work_data>" + body.replace("<", "\\u003c").replace(">", "\\u003e") + "</work_data>"
        if len(text) <= limit or not kept:
            return text
        kept.pop()


def _kept_ids(text: str) -> set[int]:
    """Return the work ids inside a rendered tool result (those the model actually saw)."""
    body = text.removeprefix("<work_data>").removesuffix("</work_data>")
    ids: set[int] = set()
    for item in json.loads(body)["results"]:
        ids.add(item["id"])
        ids.update(link["id"] for link in item.get("links", []))
    return ids


# The turn ---------------------------------------------------------------------------------------


class _Turn:
    """State for one user turn: which works the tools have returned, and why."""

    def __init__(self, conn: sqlite3.Connection, persona_id: int) -> None:
        self.conn = conn
        self.persona_id = persona_id
        self.seen: dict[int, list[str]] = {}  # work id -> reasons, for ids the model has seen

    def run_tool(self, name: str, args: Any) -> tuple[str, bool]:
        """Run one tool and return (result text for the model, is_error)."""
        handler = _HANDLERS.get(name)
        if handler is None:
            return f"unknown tool {name!r}", True
        try:
            items = handler(self.conn, self.persona_id, args if isinstance(args, dict) else {})
        except ToolError as exc:
            return str(exc), True
        text = render_tool_result(items)
        shown = _kept_ids(text)
        for item in items:
            if item["id"] in shown:
                self.seen.setdefault(item["id"], item.get("reasons", []))
            for link in item.get("links", []):
                if link["id"] in shown:
                    self.seen.setdefault(link["id"], [link["how"]])
        return text, False

    def present(self, args: Any) -> tuple[list[Card], list[int]]:
        """Validate a present_recommendations call: return (cards to show, ids dropped)."""
        args = args if isinstance(args, dict) else {}
        raw = args.get("work_ids")
        blurbs = args.get("blurbs") if isinstance(args.get("blurbs"), dict) else {}
        ids: list[int] = []
        for value in raw if isinstance(raw, list) else []:
            if isinstance(value, int) and not isinstance(value, bool) and value not in ids:
                ids.append(value)
        dropped = [i for i in ids if i not in self.seen]
        if dropped:
            logger.warning("agent presented ids no tool returned this turn; dropped %s", dropped)
        ids = [i for i in ids if i in self.seen][:AGENT_MAX_PRESENTED]
        works = browse.works_by_ids(self.conn, ids)
        cards = [
            Card(
                i,
                str(blurbs.get(str(i)) or blurbs.get(i) or "").strip(),
                list(self.seen[i]),
            )
            for i in ids
            if i in works
        ]
        return cards, dropped


def default_reply(conn: sqlite3.Connection, cards: list[Card]) -> str:
    """Return a reply for when the model showed cards but wrote no text: the split, and a nudge."""
    works = browse.works_by_ids(conn, [c.work_id for c in cards])
    counts = {"movie": 0, "book": 0}
    for card in cards:
        if card.work_id in works:
            counts[works[card.work_id].media_type] += 1
    parts = [f"{n} {name}{'' if n == 1 else 's'}" for name, n in counts.items() if n]
    return (
        f"Here {'is' if len(cards) == 1 else 'are'} {' and '.join(parts)} for that vibe. "
        "Want it darker, lighter or in just one medium?"
    )


def _blocks(content: Any) -> list[dict[str, Any]]:
    """Convert a response's content blocks to plain dicts that can be sent back to the API."""
    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text" and block.text:
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
    return blocks


def _failure(message: str, error: str, tools: list[str]) -> TurnResult:
    return TurnResult(text=message, tools_used=tools, error=error)


def run_turn(
    conn: sqlite3.Connection,
    persona_id: int,
    history: list[dict[str, str]],
    user_message: str,
    *,
    on_text: Callable[[str], None] | None = None,
    on_tool: Callable[[str], None] | None = None,
) -> TurnResult:
    """Answer one user message. `history` is earlier {"role", "content"} text messages.

    `on_text` receives streamed text as it arrives (it may be discarded if the model then calls a
    tool); `on_tool` receives each tool name as it runs. Never raises for API or budget problems:
    those come back as a TurnResult with `error` set and a message fit to show the user.
    """
    system = build_system_prompt(conn, persona_id)
    messages: list[dict[str, Any]] = [
        {"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY_MESSAGES:]
    ]
    messages.append({"role": "user", "content": user_message})
    turn = _Turn(conn, persona_id)
    tools_used: list[str] = []
    try:
        for _ in range(AGENT_MAX_ROUNDS):
            response = llm.complete_stream(
                conn,
                purpose="chat_agent",
                model=AGENT_MODEL,
                messages=messages,
                max_tokens=AGENT_MAX_TOKENS,
                system=system,
                tools=TOOLS,
                thinking=llm.THINKING_OFF,
                on_text=on_text,
            )
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            uses = [b for b in response.content if b.type == "tool_use"]
            if not uses:
                return TurnResult(text=text or EMPTY_REPLY, tools_used=tools_used)

            results: list[dict[str, Any]] = []
            for use in (u for u in uses if u.name != PRESENT):
                tools_used.append(use.name)
                if on_tool:
                    on_tool(use.name)
                content, is_error = turn.run_tool(use.name, use.input)
                results.append(_tool_result(use.id, content, is_error))
            presents = [u for u in uses if u.name == PRESENT]
            for number, use in enumerate(presents):
                tools_used.append(use.name)
                if number == 0:
                    cards, dropped = turn.present(use.input)
                    if cards:
                        return TurnResult(
                            text=text or default_reply(conn, cards),
                            cards=cards,
                            dropped_ids=dropped,
                            tools_used=tools_used,
                        )
                    note = (
                        "None of those ids came from tool results this turn. Call a search or "
                        "recommend tool first, then present only ids it returned."
                    )
                else:
                    note = f"Call {PRESENT} only once."
                results.append(_tool_result(use.id, note, True))
            messages.append({"role": "assistant", "content": _blocks(response.content)})
            messages.append({"role": "user", "content": results})
        logger.warning("agent used all %d rounds without an answer", AGENT_MAX_ROUNDS)
        return _failure(TANGLED_REPLY, "max_rounds", tools_used)
    except llm.BudgetExceeded as exc:
        logger.warning("chat blocked by the daily budget: %s", exc)
        return _failure(
            "I've hit today's spending limit for the language model, so chat is paused. "
            "Browsing still works.",
            "budget",
            tools_used,
        )
    except (anthropic.APIError, RuntimeError) as exc:
        logger.error("chat model call failed: %s: %s", type(exc).__name__, exc)
        return _failure(
            "I couldn't reach the language model just now (check the Anthropic key and credit "
            "balance in `.env`). Browsing still works.",
            type(exc).__name__,
            tools_used,
        )


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        result["is_error"] = True
    return result


def history_entry(conn: sqlite3.Connection, result: TurnResult) -> str:
    """Return the assistant text to keep in the model's history, noting which cards were shown.

    Without the note, a follow-up like "something shorter than those" would not know which works
    "those" are. Titles are catalog text, so they are only ever sent back inside the history of the
    next request as plain context, never as instructions.
    """
    if not result.cards:
        return result.text
    works = browse.works_by_ids(conn, [c.work_id for c in result.cards])
    shown = ", ".join(
        f"{works[c.work_id].title} ({works[c.work_id].media_type})"
        for c in result.cards
        if c.work_id in works
    )
    return f"{result.text}\n[Cards shown to the user: {shown}]"


# Chat log ---------------------------------------------------------------------------------------


def log_message(
    conn: sqlite3.Connection,
    persona_id: int,
    session_id: str,
    role: str,
    content: str,
    recommended_ids: list[int] | None = None,
) -> None:
    """Append one message to `chat_messages` (ids of the shown cards go in `recommended_ids`)."""
    conn.execute(
        "INSERT INTO chat_messages (persona_id, session_id, role, content, recommended_ids) "
        "VALUES (?, ?, ?, ?, ?)",
        (persona_id, session_id, role, content, json.dumps(recommended_ids or [])),
    )
    conn.commit()
