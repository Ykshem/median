"""Thumbs up / down on recommendations. Each vote moves the persona's taste vector.

One vote per (persona, work): voting again replaces the old vote, and `None` clears it, so
clicking a button repeatedly can't stack weight (ARCHITECTURE.md section 5).
"""

from __future__ import annotations

import sqlite3

from median import taste

SIGNALS = ("up", "down")


def get_signal(conn: sqlite3.Connection, persona_id: int, work_id: int) -> str | None:
    """Return the persona's current vote on this work ('up' or 'down'), or None."""
    row = conn.execute(
        "SELECT signal FROM rec_feedback WHERE persona_id = ? AND work_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (persona_id, work_id),
    ).fetchone()
    return row["signal"] if row else None


def set_signal(
    conn: sqlite3.Connection,
    persona_id: int,
    work_id: int,
    signal: str | None,
    seed_id: int | None = None,
) -> None:
    """Record the vote (replacing any earlier one), or clear it with None; recompute and commit."""
    if signal is not None and signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS} or None, got {signal!r}")
    conn.execute(
        "DELETE FROM rec_feedback WHERE persona_id = ? AND work_id = ?", (persona_id, work_id)
    )
    if signal is not None:
        conn.execute(
            "INSERT INTO rec_feedback (persona_id, work_id, seed_id, signal) VALUES (?, ?, ?, ?)",
            (persona_id, work_id, seed_id, signal),
        )
    taste.recompute(conn, persona_id)
    conn.commit()
