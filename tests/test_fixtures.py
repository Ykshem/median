from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from median.config import EMBED_DIMS
from median.db import init_db, vec_to_blob
from median.models import VibeCard
from tests.fixtures.catalog import cluster_centroid, load_mini_catalog, work_vector

MEDIUM_WORDS = re.compile(
    r"\b(film|films|movie|movies|novel|novels|book|books|reader|readers|viewer|viewers|"
    r"chapter|chapters|pages|cinematography|prose|runtime)\b",
    re.IGNORECASE,
)


def _knn(db: sqlite3.Connection, work_id: int, k: int, media: str | None = None) -> list[int]:
    blob = db.execute("SELECT embedding FROM vec_vibes WHERE work_id = ?", (work_id,)).fetchone()[0]
    extra = "AND media_type = ?" if media else ""
    params = (blob, k, media) if media else (blob, k)
    sql = (
        f"SELECT work_id FROM vec_vibes WHERE embedding MATCH ? AND k = ? {extra} ORDER BY distance"
    )
    return [r["work_id"] for r in db.execute(sql, params)]


def test_catalog_shape(db: sqlite3.Connection, mini_catalog: dict[str, Any]) -> None:
    counts = dict(
        db.execute("SELECT media_type, COUNT(*) FROM works GROUP BY media_type").fetchall()
    )
    assert counts == {"movie": 10, "book": 10}
    assert db.execute("SELECT COUNT(*) FROM vibe_cards").fetchone()[0] == 20
    assert db.execute("SELECT COUNT(*) FROM vec_vibes").fetchone()[0] == 20
    types = {r[0] for r in db.execute("SELECT link_type FROM work_links")}
    assert types == {"adaptation_of", "same_franchise", "same_creator"}


def test_every_card_validates_and_card_text_matches(
    db: sqlite3.Connection, mini_catalog: dict[str, Any]
) -> None:
    for row in db.execute("SELECT card_json, card_text FROM vibe_cards"):
        card = VibeCard.model_validate_json(row["card_json"])
        assert row["card_text"] == card.to_text()


def test_cards_follow_the_template_rules(
    db: sqlite3.Connection, mini_catalog: dict[str, Any]
) -> None:
    """No medium words, no titles, no creator names (ARCHITECTURE.md section 4)."""
    for work in mini_catalog["works"]:
        text = db.execute(
            "SELECT card_text FROM vibe_cards WHERE work_id = ?", (work["id"],)
        ).fetchone()[0]
        assert not MEDIUM_WORDS.search(text), f"medium word in card for {work['title']}"
    all_text = " ".join(r[0] for r in db.execute("SELECT card_text FROM vibe_cards")).lower()
    for work in mini_catalog["works"]:
        for needle in [work["title"], *work["creators"]]:
            assert not re.search(rf"\b{re.escape(needle.lower())}\b", all_text), needle


def test_vectors_have_full_dimension_and_unit_length(
    db: sqlite3.Connection, mini_catalog: dict[str, Any]
) -> None:
    vec = work_vector(mini_catalog["clusters"]["haunted"], 1)
    assert len(vec) == EMBED_DIMS
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-9


def test_vectors_are_deterministic(tmp_path: Path) -> None:
    blobs = []
    for name in ("a.db", "b.db"):
        conn = init_db(tmp_path / name)
        load_mini_catalog(conn)
        blobs.append(conn.execute("SELECT embedding FROM vec_vibes ORDER BY work_id").fetchall())
        conn.close()
    assert [bytes(r[0]) for r in blobs[0]] == [bytes(r[0]) for r in blobs[1]]


def test_nearest_neighbour_shares_the_cluster(
    db: sqlite3.Connection, mini_catalog: dict[str, Any]
) -> None:
    cluster = {w["id"]: w["cluster"] for w in mini_catalog["works"]}
    for work_id in cluster:
        nearest = _knn(db, work_id, k=2)
        other = next(i for i in nearest if i != work_id)
        assert cluster[other] == cluster[work_id], f"work {work_id} -> {other}"


def test_cross_media_knn_returns_the_other_medium(
    db: sqlite3.Connection, mini_catalog: dict[str, Any]
) -> None:
    # Same-cluster order is decided by seeded noise, so compare sets, not ranks.
    # Shining film (1): its two nearest books are the two haunted books (11, 12).
    assert set(_knn(db, 1, k=2, media="book")) == {11, 12}
    # Orient Express novel (14): its two nearest films are the two cozy-mystery films (3, 4).
    assert set(_knn(db, 14, k=2, media="movie")) == {3, 4}


def test_jurassic_pair_is_linked_but_split_across_clusters(
    db: sqlite3.Connection, mini_catalog: dict[str, Any]
) -> None:
    by_id = {w["id"]: w for w in mini_catalog["works"]}
    assert by_id[9]["cluster"] != by_id[19]["cluster"]
    link = db.execute("SELECT 1 FROM work_links WHERE from_work = 9 AND to_work = 19").fetchone()
    assert link is not None
    # Vibe search alone still lands in the right neighbourhood (adventure sits near sci-fi).
    assert set(_knn(db, 9, k=2, media="book")) == {19, 20}


def test_hobbit_pair_is_deliberately_unlinked(
    db: sqlite3.Connection, mini_catalog: dict[str, Any]
) -> None:
    row = db.execute(
        "SELECT 1 FROM work_links WHERE (from_work = 6 AND to_work = 16) "
        "OR (from_work = 16 AND to_work = 6)"
    ).fetchone()
    assert row is None
    # Vibe similarity still surfaces the book: the two nearest books are the two epic-cluster books.
    assert set(_knn(db, 6, k=2, media="book")) == {15, 16}


def test_helpers_are_pure_functions_of_their_inputs() -> None:
    weights = json.loads(Path("tests/fixtures/mini_catalog.json").read_text())["clusters"]["epic"]
    assert cluster_centroid(weights) == cluster_centroid(weights)
    assert vec_to_blob(work_vector(weights, 5)) == vec_to_blob(work_vector(weights, 5))
