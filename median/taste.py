"""Persona taste vectors: a weighted mean of the vibe vectors of works the persona has reacted to.

Signals (ARCHITECTURE.md section 5):
  rating r (1-10)         weight (r - 5.5) / 4.5, so -1..+1
  shelf "favorite"        +1.0 (a favorite that is also rated uses the larger weight, not both)
  thumbs up / down        +0.5 / -0.5 per feedback row
The vector is normalised, and stays NULL until there are at least TASTE_MIN_SIGNALS signals on works
that have a vibe vector. Functions here do not commit; the caller owns the transaction.
"""

from __future__ import annotations

import array
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from median.config import (
    TASTE_FAVORITE_WEIGHT,
    TASTE_FEEDBACK_WEIGHT,
    TASTE_MIN_SIGNALS,
    TASTE_RATING_MIDPOINT,
    TASTE_RATING_SPAN,
)
from median.db import vec_to_blob

MIN_NORM = 1e-9


@dataclass(frozen=True)
class Signals:
    """The persona's net weight per work, and how many separate signals produced them."""

    weights: dict[int, float]
    count: int

    def liked(self) -> list[int]:
        """Return work ids with a positive net weight, strongest first."""
        positive = [(w, i) for i, w in self.weights.items() if w > 0]
        return [i for _, i in sorted(positive, key=lambda t: (-t[0], t[1]))]

    def disliked(self) -> list[int]:
        """Return work ids with a negative net weight, most disliked first."""
        negative = [(w, i) for i, w in self.weights.items() if w < 0]
        return [i for _, i in sorted(negative, key=lambda t: (t[0], t[1]))]


def rating_weight(rating: int) -> float:
    """Return the weight of a 1-10 rating: -1 for a 1 up to +1 for a 10."""
    return (rating - TASTE_RATING_MIDPOINT) / TASTE_RATING_SPAN


def decode(blob: bytes) -> list[float]:
    """Decode a float32 vector blob."""
    return list(array.array("f", blob))


def normalize(vector: list[float]) -> list[float] | None:
    """Return the unit vector, or None if it is (almost) zero."""
    norm = math.sqrt(sum(x * x for x in vector))
    return None if norm < MIN_NORM else [x / norm for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two vectors (0 if either is zero)."""
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na < MIN_NORM or nb < MIN_NORM:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def weighted_direction(items: Iterable[tuple[list[float], float]]) -> list[float] | None:
    """Return the normalised weighted sum of vectors, or None if nothing (or zero) is left.

    Negative weights push the result away from those vectors.
    """
    total: list[float] | None = None
    for vector, weight in items:
        if total is None:
            total = [0.0] * len(vector)
        for i, x in enumerate(vector):
            total[i] += weight * x
    return None if total is None else normalize(total)


def collect_signals(conn: sqlite3.Connection, persona_id: int) -> Signals:
    """Gather the persona's signals from shelf ratings, favorites and recommendation feedback.

    Only works that have a vibe vector count, since the others can't move the taste vector.
    """
    weights: dict[int, float] = {}
    count = 0
    for row in conn.execute(
        "SELECT s.work_id, s.shelf, s.rating FROM shelf_items s "
        "WHERE s.persona_id = ? AND s.work_id IN (SELECT work_id FROM vec_vibes)",
        (persona_id,),
    ):
        weight: float | None = rating_weight(row["rating"]) if row["rating"] is not None else None
        if row["shelf"] == "favorite":
            weight = TASTE_FAVORITE_WEIGHT if weight is None else max(weight, TASTE_FAVORITE_WEIGHT)
        if weight is not None:
            weights[row["work_id"]] = weights.get(row["work_id"], 0.0) + weight
            count += 1
    for row in conn.execute(
        "SELECT f.work_id, f.signal FROM rec_feedback f "
        "WHERE f.persona_id = ? AND f.work_id IN (SELECT work_id FROM vec_vibes)",
        (persona_id,),
    ):
        sign = TASTE_FEEDBACK_WEIGHT if row["signal"] == "up" else -TASTE_FEEDBACK_WEIGHT
        weights[row["work_id"]] = weights.get(row["work_id"], 0.0) + sign
        count += 1
    return Signals(weights, count)


def _vectors(conn: sqlite3.Connection, work_ids: list[int]) -> dict[int, list[float]]:
    if not work_ids:
        return {}
    marks = ",".join("?" * len(work_ids))
    rows = conn.execute(
        f"SELECT work_id, embedding FROM vec_vibes WHERE work_id IN ({marks})",  # noqa: S608
        work_ids,
    )
    return {r["work_id"]: decode(r["embedding"]) for r in rows}


def compute_taste_vector(conn: sqlite3.Connection, persona_id: int) -> list[float] | None:
    """Return the persona's taste vector, or None when there are too few signals."""
    signals = collect_signals(conn, persona_id)
    if signals.count < TASTE_MIN_SIGNALS:
        return None
    vectors = _vectors(conn, list(signals.weights))
    return weighted_direction((vectors[i], w) for i, w in signals.weights.items() if i in vectors)


def get_taste_vector(conn: sqlite3.Connection, persona_id: int) -> list[float] | None:
    """Return the stored taste vector, or None."""
    row = conn.execute("SELECT taste_vector FROM personas WHERE id = ?", (persona_id,)).fetchone()
    return decode(row["taste_vector"]) if row and row["taste_vector"] else None


def recompute(conn: sqlite3.Connection, persona_id: int) -> bool:
    """Recompute and store one persona's taste vector. Returns True if it now has one.

    Call after any rating, shelf or feedback change. Does not commit.
    """
    vector = compute_taste_vector(conn, persona_id)
    conn.execute(
        "UPDATE personas SET taste_vector = ?, updated_at = ? WHERE id = ?",
        (
            vec_to_blob(vector) if vector else None,
            datetime.now(UTC).isoformat(timespec="seconds"),
            persona_id,
        ),
    )
    return vector is not None


def recompute_all(conn: sqlite3.Connection) -> int:
    """Recompute every persona's taste vector; return how many have one afterwards."""
    ids = [r["id"] for r in conn.execute("SELECT id FROM personas ORDER BY id")]
    return sum(recompute(conn, persona_id) for persona_id in ids)


@dataclass(frozen=True)
class TasteProfile:
    """What a persona's taste vector looks like in words, for the Taste page."""

    signals: int  # ratings, favorites and thumbs that count (only works with a vibe vector)
    has_vector: bool
    themes: list[tuple[str, int]]  # (theme, how many nearby cards share it), most common first
    moods: list[tuple[str, int]]
    nearest: list[int]  # work ids of the cards closest to the taste vector, nearest first
    liked: list[int]
    disliked: list[int]


def profile(
    conn: sqlite3.Connection, persona_id: int, neighbours: int = 30, top: int = 8
) -> TasteProfile:
    """Describe the persona's taste by the themes and moods of the cards nearest its taste vector.

    Without a taste vector (too few signals, or no embeddings yet) the term lists are empty.
    """
    signals = collect_signals(conn, persona_id)
    vector = get_taste_vector(conn, persona_id)
    themes: Counter[str] = Counter()
    moods: Counter[str] = Counter()
    nearest: list[int] = []
    if vector is not None:
        rows = conn.execute(
            "SELECT work_id FROM vec_vibes WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_to_blob(vector), neighbours),
        ).fetchall()
        nearest = [r["work_id"] for r in rows]
        for work_id in nearest:
            card = conn.execute(
                "SELECT card_json FROM vibe_cards WHERE work_id = ?", (work_id,)
            ).fetchone()
            if card is None:
                continue
            data = json.loads(card["card_json"])
            themes.update(t.lower() for t in data.get("themes", []))
            moods.update(m.lower() for m in data.get("mood", []))
    return TasteProfile(
        signals=signals.count,
        has_vector=vector is not None,
        themes=themes.most_common(top),
        moods=moods.most_common(top),
        nearest=nearest,
        liked=signals.liked(),
        disliked=signals.disliked(),
    )
