"""Embed each vibe card's text and fill the vec_vibes index (OpenAI text-embedding-3-small)."""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass

from median import llm
from median.config import EMBED_DIMS, EMBED_MODEL, VIBE_PROMPT_VERSION
from median.db import get_meta, set_meta, vec_to_blob

logger = logging.getLogger(__name__)

PURPOSE = "embed"
CHUNK = 256  # one request and one commit per chunk, so an interrupted run resumes
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class EmbedReport:
    embedded: int
    already_done: int


def pending_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return (id, media_type, card_text) for current-version cards that have no vector yet."""
    return conn.execute(
        "SELECT w.id, w.media_type, c.card_text FROM vibe_cards c JOIN works w ON w.id = c.work_id "
        "WHERE c.prompt_version = ? AND c.work_id NOT IN (SELECT work_id FROM vec_vibes) "
        "ORDER BY w.id",
        (VIBE_PROMPT_VERSION,),
    ).fetchall()


def estimate_usd(rows: list[sqlite3.Row]) -> float:
    """Estimate the embedding cost from the text length (about 4 characters per token)."""
    tokens = sum(math.ceil(len(r["card_text"]) / CHARS_PER_TOKEN) for r in rows)
    return llm.cost_usd(EMBED_MODEL, tokens, 0)


def embed_cards(conn: sqlite3.Connection) -> EmbedReport:
    """Embed every card that lacks a vector; vectors from an older prompt version are discarded.

    Cosine KNN needs every vector to come from the same template, so when the cards were
    regenerated under a new `VIBE_PROMPT_VERSION` the whole index is rebuilt.
    """
    embedded_version = get_meta(conn, "embedded_prompt_version")
    if embedded_version not in (None, VIBE_PROMPT_VERSION):
        logger.info(
            "Prompt version changed (%s -> %s): re-embedding all",
            embedded_version,
            VIBE_PROMPT_VERSION,
        )
        conn.execute("DELETE FROM vec_vibes")
        conn.commit()
    already = conn.execute("SELECT COUNT(*) FROM vec_vibes").fetchone()[0]
    rows = pending_rows(conn)
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        vectors = llm.embed(conn, [r["card_text"] for r in chunk], purpose=PURPOSE)
        for row, vector in zip(chunk, vectors, strict=True):
            conn.execute(
                "INSERT INTO vec_vibes (work_id, embedding, media_type) VALUES (?, ?, ?)",
                (row["id"], vec_to_blob(vector), row["media_type"]),
            )
        conn.commit()
        logger.info("  embedded %d / %d", min(start + CHUNK, len(rows)), len(rows))
    set_meta(conn, "embedding_model", EMBED_MODEL)
    set_meta(conn, "embedding_dims", str(EMBED_DIMS))
    set_meta(conn, "embedded_prompt_version", VIBE_PROMPT_VERSION)
    conn.commit()
    return EmbedReport(embedded=len(rows), already_done=already)
