"""Overlap scoring: find works in the other medium that a work (or a persona's taste) points to.

    score = 0.45 * link + 0.35 * vibe_similarity + 0.15 * taste_similarity + 0.05 * popularity

Candidates come from two places (ARCHITECTURE.md section 5):
  links : verified `work_links` touching the seed in either direction, plus one hop through a shared
          franchise (a series neighbour's own links count at a reduced strength);
  vibe  : the nearest neighbours of the seed's vector in `vec_vibes`, filtered to the target medium.
Then the seed and anything on the persona's shelves are dropped, at most MAX_PER_FRANCHISE results
may come from one franchise, and each result gets plain-language reasons. Reads only; never writes.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field

from median import llm, taste
from median.config import (
    KNN_CANDIDATES,
    LINK_STRENGTH,
    MAX_PER_FRANCHISE,
    MAX_REASON_THEMES,
    SCORE_WEIGHTS,
    TASTE_REASON_MIN,
)
from median.db import vec_to_blob
from median.models import Recommendation, VibeCard

TARGET_MEDIA = ("movie", "book", "tv", "music", "any")
PERSONA_THEME_POOL = 12  # most frequent themes/moods among a persona's liked works


@dataclass
class _Candidate:
    work_id: int
    link: float = 0.0
    link_reasons: list[tuple[float, str, int]] = field(default_factory=list)  # (strength, kind, id)
    vibe: float = 0.0
    taste: float = 0.0
    popularity: float = 0.0


# Small queries ---------------------------------------------------------------------------------


def _check_target(target_media: str, k: int) -> None:
    if target_media not in TARGET_MEDIA:
        raise ValueError(f"target_media must be one of {TARGET_MEDIA}, not {target_media!r}")
    if k < 1:
        raise ValueError("k must be at least 1")


def _embedding(conn: sqlite3.Connection, work_id: int) -> bytes | None:
    row = conn.execute("SELECT embedding FROM vec_vibes WHERE work_id = ?", (work_id,)).fetchone()
    return row["embedding"] if row else None


def _knn(
    conn: sqlite3.Connection, query: bytes, limit: int, media: str | None
) -> list[tuple[int, float]]:
    """Return (work_id, 1 - distance) for the nearest vectors, best first."""
    sql = "SELECT work_id, distance FROM vec_vibes WHERE embedding MATCH ? AND k = ?"
    params: list[object] = [query, limit]
    if media:
        sql += " AND media_type = ?"
        params.append(media)
    rows = conn.execute(sql + " ORDER BY distance", params).fetchall()
    return [(r["work_id"], max(0.0, 1.0 - r["distance"])) for r in rows]


def _similarity(conn: sqlite3.Connection, query: bytes, work_id: int) -> float:
    """Return the cosine similarity (0..1) between `query` and a work's vector; 0 if it has none."""
    row = conn.execute(
        "SELECT vec_distance_cosine(?, embedding) AS d FROM vec_vibes WHERE work_id = ?",
        (query, work_id),
    ).fetchone()
    return 0.0 if row is None else max(0.0, 1.0 - row["d"])


def _work_info(conn: sqlite3.Connection, ids: list[int]) -> dict[int, sqlite3.Row]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, title, media_type, popularity FROM works WHERE id IN ({marks})",  # noqa: S608
        ids,
    )
    return {r["id"]: r for r in rows}


def _card(conn: sqlite3.Connection, work_id: int) -> VibeCard | None:
    row = conn.execute("SELECT card_json FROM vibe_cards WHERE work_id = ?", (work_id,)).fetchone()
    return VibeCard.model_validate_json(row["card_json"]) if row else None


def _persona(conn: sqlite3.Connection, persona_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, name, taste_vector FROM personas WHERE id = ?", (persona_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown persona {persona_id}")
    return row


def _shelved(conn: sqlite3.Connection, persona_id: int | None) -> set[int]:
    if persona_id is None:
        return set()
    rows = conn.execute("SELECT work_id FROM shelf_items WHERE persona_id = ?", (persona_id,))
    return {r["work_id"] for r in rows}


# Links ----------------------------------------------------------------------------------------


def _verified_links(conn: sqlite3.Connection, work_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT from_work, to_work, link_type, confidence FROM work_links "
        "WHERE verified = 1 AND (from_work = ? OR to_work = ?)",
        (work_id, work_id),
    ).fetchall()


def _link_candidates(conn: sqlite3.Connection, seed_id: int) -> dict[int, _Candidate]:
    """Return candidates reached by verified links, with their strength and why.

    Strength is LINK_STRENGTH[type] x confidence. A series neighbour's own links count too, but
    scaled by the same-franchise strength (one hop only).
    """
    found: dict[int, _Candidate] = {}

    def add(work_id: int, strength: float, kind: str, ref: int) -> None:
        cand = found.setdefault(work_id, _Candidate(work_id))
        cand.link = max(cand.link, strength)
        cand.link_reasons.append((strength, kind, ref))

    series_neighbours: list[int] = []
    for link in _verified_links(conn, seed_id):
        other = link["to_work"] if link["from_work"] == seed_id else link["from_work"]
        strength = LINK_STRENGTH[link["link_type"]] * link["confidence"]
        if link["link_type"] == "adaptation_of":
            # from_work is adapted from to_work
            kind = "source_of_seed" if link["from_work"] == seed_id else "adapted_from_seed"
        elif link["link_type"] == "same_franchise":
            kind = "same_series"
            series_neighbours.append(other)
        else:
            kind = "same_creator"
        add(other, strength, kind, seed_id)

    hop = LINK_STRENGTH["same_franchise"]
    for neighbour in series_neighbours:
        for link in _verified_links(conn, neighbour):
            other = link["to_work"] if link["from_work"] == neighbour else link["from_work"]
            if other in (seed_id, neighbour):
                continue
            add(
                other,
                LINK_STRENGTH[link["link_type"]] * link["confidence"] * hop,
                "series_hop",
                neighbour,
            )
    return found


def _franchise_of(conn: sqlite3.Connection) -> dict[int, int]:
    """Map each work in a series to its franchise id (the smallest work id in its group)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in conn.execute(
        "SELECT from_work, to_work FROM work_links "
        "WHERE link_type = 'same_franchise' AND verified = 1"
    ):
        a, b = find(row["from_work"]), find(row["to_work"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {work_id: find(work_id) for work_id in list(parent)}


# Ranking --------------------------------------------------------------------------------------


def _score(cand: _Candidate) -> float:
    return (
        SCORE_WEIGHTS["link"] * cand.link
        + SCORE_WEIGHTS["vibe"] * cand.vibe
        + SCORE_WEIGHTS["taste"] * cand.taste
        + SCORE_WEIGHTS["popularity"] * cand.popularity
    )


def _diversify(ranked: list[_Candidate], franchise_of: dict[int, int], k: int) -> list[_Candidate]:
    """Take the best k, allowing at most MAX_PER_FRANCHISE from any one franchise."""
    per_franchise: Counter[int] = Counter()
    chosen: list[_Candidate] = []
    for cand in ranked:
        franchise = franchise_of.get(cand.work_id)
        if franchise is not None:
            if per_franchise[franchise] >= MAX_PER_FRANCHISE:
                continue
            per_franchise[franchise] += 1
        chosen.append(cand)
        if len(chosen) == k:
            break
    return chosen


def _rank(cands: list[_Candidate]) -> list[_Candidate]:
    return sorted(cands, key=lambda c: (-_score(c), c.work_id))


# Reasons --------------------------------------------------------------------------------------


def _shared_terms(pool: list[str], card: VibeCard | None) -> list[str]:
    """Return themes and moods from `pool` that the card also has, themes first, in pool order."""
    if card is None:
        return []
    have = {t.lower() for t in [*card.themes, *card.mood]}
    seen: dict[str, None] = {}
    for term in pool:
        if term.lower() in have:
            seen.setdefault(term)
    return list(seen)[:MAX_REASON_THEMES]


def _card_pool(card: VibeCard | None) -> list[str]:
    return [] if card is None else [*card.themes, *card.mood]


def _link_reason(kind: str, seed_title: str, ref_title: str) -> str:
    return {
        "source_of_seed": f"Source of {seed_title}",
        "adapted_from_seed": f"Adapted from {seed_title}",
        "same_series": "Same series",
        "series_hop": f"Linked to {ref_title}, from the same series",
        "same_creator": "Shared creator",
    }[kind]


def _build_reasons(
    cand: _Candidate,
    *,
    titles: dict[int, str],
    seed_title: str,
    pool: list[str],
    conn: sqlite3.Connection,
    persona_name: str | None,
    mood: bool = False,
) -> list[str]:
    reasons: list[str] = []
    for _, kind, ref in sorted(cand.link_reasons, key=lambda t: -t[0]):
        text = _link_reason(kind, seed_title, titles.get(ref, "a related work"))
        if text not in reasons:
            reasons.append(text)
    shared = _shared_terms(pool, _card(conn, cand.work_id))
    if shared:
        reasons.append("Shared themes: " + ", ".join(shared))
    elif cand.vibe > 0 and not reasons:
        reasons.append("Similar vibe")
    if mood:
        reasons.append("Matches your mood request")
    if persona_name and cand.taste >= TASTE_REASON_MIN:
        reasons.append(f"Fits {persona_name}'s taste")
    return reasons


def _finish(chosen: list[_Candidate], reasons: dict[int, list[str]]) -> list[Recommendation]:
    return [
        Recommendation(
            work_id=c.work_id,
            score=round(_score(c), 6),
            reasons=reasons[c.work_id],
            link=c.link,
            vibe=c.vibe,
            taste=c.taste,
            popularity=c.popularity,
        )
        for c in chosen
    ]


# Public API -----------------------------------------------------------------------------------


def overlap(
    conn: sqlite3.Connection,
    seed_id: int,
    target_media: str = "any",
    persona_id: int | None = None,
    k: int = 10,
) -> list[Recommendation]:
    """Return up to k works related to `seed_id`, best first, each with reasons.

    target_media: 'movie', 'book', ... or 'any'. With a persona, works already on their shelves are
    left out and their taste vector adds to the score. Raises ValueError for an unknown work or
    persona.
    """
    _check_target(target_media, k)
    seed = _work_info(conn, [seed_id]).get(seed_id)
    if seed is None:
        raise ValueError(f"unknown work {seed_id}")
    persona = _persona(conn, persona_id) if persona_id is not None else None
    taste_blob = persona["taste_vector"] if persona else None
    media = None if target_media == "any" else target_media

    candidates = _link_candidates(conn, seed_id)
    seed_blob = _embedding(conn, seed_id)
    if seed_blob is not None:
        for work_id, sim in _knn(conn, seed_blob, KNN_CANDIDATES + 1, media):
            candidates.setdefault(work_id, _Candidate(work_id)).vibe = sim
    candidates.pop(seed_id, None)
    for work_id in _shelved(conn, persona_id):
        candidates.pop(work_id, None)

    info = _work_info(conn, list(candidates))
    for work_id in list(candidates):
        row = info.get(work_id)
        if row is None or (media and row["media_type"] != media):
            del candidates[work_id]
    for cand in candidates.values():
        if cand.vibe == 0.0 and seed_blob is not None:
            cand.vibe = _similarity(conn, seed_blob, cand.work_id)
        if taste_blob is not None:
            cand.taste = _similarity(conn, taste_blob, cand.work_id)
        cand.popularity = info[cand.work_id]["popularity"] or 0.0

    chosen = _diversify(_rank(list(candidates.values())), _franchise_of(conn), k)
    titles = {seed_id: seed["title"], **{i: r["title"] for i, r in info.items()}}
    ref_ids = [ref for c in chosen for _, _, ref in c.link_reasons if ref not in titles]
    titles.update({i: r["title"] for i, r in _work_info(conn, ref_ids).items()})
    pool = _card_pool(_card(conn, seed_id))
    reasons = {
        c.work_id: _build_reasons(
            c,
            titles=titles,
            seed_title=seed["title"],
            pool=pool,
            conn=conn,
            persona_name=persona["name"] if persona else None,
        )
        for c in chosen
    }
    return _finish(chosen, reasons)


def _persona_theme_pool(conn: sqlite3.Connection, persona_id: int) -> list[str]:
    """Return the themes and moods most common among the works the persona likes."""
    counts: Counter[str] = Counter()
    for work_id in taste.collect_signals(conn, persona_id).liked():
        for term in _card_pool(_card(conn, work_id)):
            counts[term.lower()] += 1
    return [term for term, _ in counts.most_common(PERSONA_THEME_POOL)]


def for_persona(
    conn: sqlite3.Connection,
    persona_id: int,
    target_media: str = "any",
    mood_vector: list[float] | None = None,
    k: int = 10,
) -> list[Recommendation]:
    """Return up to k works that fit a persona's taste, optionally steered by a mood vector.

    The query is the persona's taste vector, blended equally with `mood_vector` (an embedding of the
    user's request; see `embed_query`) when given. With neither, the most popular unshelved works
    are returned. Works already on the persona's shelves are left out.
    """
    _check_target(target_media, k)
    persona = _persona(conn, persona_id)
    media = None if target_media == "any" else target_media
    taste_vec = taste.get_taste_vector(conn, persona_id)
    taste_blob = persona["taste_vector"] if taste_vec else None
    parts = [
        v
        for v in (
            taste.normalize(taste_vec) if taste_vec else None,
            taste.normalize(mood_vector) if mood_vector else None,
        )
        if v
    ]
    query = taste.weighted_direction((v, 1.0) for v in parts) if parts else None
    shelved = _shelved(conn, persona_id)

    candidates: dict[int, _Candidate] = {}
    if query is not None:
        query_blob = vec_to_blob(query)
        for work_id, sim in _knn(conn, query_blob, KNN_CANDIDATES + len(shelved), media):
            if work_id not in shelved:
                candidates[work_id] = _Candidate(work_id, vibe=sim)
    else:
        sql = "SELECT id FROM works WHERE popularity IS NOT NULL"
        rows = conn.execute(
            sql + (" AND media_type = ?" if media else ""), (media,) if media else ()
        )
        candidates = {r["id"]: _Candidate(r["id"]) for r in rows if r["id"] not in shelved}

    info = _work_info(conn, list(candidates))
    for work_id in list(candidates):
        if work_id not in info:
            del candidates[work_id]
    for cand in candidates.values():
        if taste_blob is not None:
            cand.taste = _similarity(conn, taste_blob, cand.work_id)
        cand.popularity = info[cand.work_id]["popularity"] or 0.0

    chosen = _diversify(_rank(list(candidates.values())), _franchise_of(conn), k)
    pool = _persona_theme_pool(conn, persona_id)
    reasons = {}
    for cand in chosen:
        reasons[cand.work_id] = _build_reasons(
            cand,
            titles={},
            seed_title="",
            pool=pool,
            conn=conn,
            persona_name=persona["name"],
            mood=mood_vector is not None,
        )
        if query is None:
            reasons[cand.work_id] = ["Popular pick"]
    return _finish(chosen, reasons)


def embed_query(conn: sqlite3.Connection, text: str) -> list[float]:
    """Embed a free-text mood request (one OpenAI call, logged and budget-checked)."""
    return llm.embed(conn, [text], purpose="mood_query")[0]
