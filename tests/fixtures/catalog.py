"""Loads the ~20-work mini catalog into a test DB, with deterministic fake vectors.

Vectors are generated, not stored: each cluster centroid is a weighted sum of random unit "axis"
vectors, and each work is its cluster centroid plus a little seeded noise. Works in one cluster are
near-identical (cosine ~0.98); unrelated clusters are near-orthogonal; clusters that share an axis
are partly similar. No network, no embeddings API.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Any

from median.config import EMBED_DIMS, EMBED_MODEL, VIBE_PROMPT_VERSION
from median.db import rebuild_fts, set_meta, vec_to_blob
from median.models import VibeCard

CATALOG_PATH = Path(__file__).parent / "mini_catalog.json"
NOISE_NORM = 0.15


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _random_unit(seed: str) -> list[float]:
    rng = random.Random(seed)  # str seeds are deterministic across runs
    return _normalize([rng.gauss(0.0, 1.0) for _ in range(EMBED_DIMS)])


def cluster_centroid(weights: dict[str, float]) -> list[float]:
    """Return the unit-length centroid for a cluster's axis weights."""
    total = [0.0] * EMBED_DIMS
    for axis, weight in weights.items():
        for i, x in enumerate(_random_unit(f"axis:{axis}")):
            total[i] += weight * x
    return _normalize(total)


def work_vector(weights: dict[str, float], work_id: int) -> list[float]:
    """Return the fake embedding for one work: its cluster centroid plus seeded noise."""
    centroid = cluster_centroid(weights)
    noise = _random_unit(f"work:{work_id}")
    return _normalize([c + NOISE_NORM * n for c, n in zip(centroid, noise, strict=True)])


def read_catalog() -> dict[str, Any]:
    """Return the parsed mini_catalog.json."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def load_mini_catalog(conn: sqlite3.Connection) -> dict[str, Any]:
    """Insert the mini catalog (works, cards, vectors, links, FTS) into `conn`; return the JSON.

    Raises pydantic.ValidationError if any vibe card breaks the VibeCard rules.
    """
    catalog = read_catalog()
    for work in catalog["works"]:
        card = VibeCard.model_validate(work["card"])
        conn.execute(
            "INSERT INTO works (id, media_type, title, year, creators, synopsis, subjects, "
            "popularity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                work["id"],
                work["media_type"],
                work["title"],
                work["year"],
                json.dumps(work["creators"]),
                work["synopsis"],
                json.dumps(work["subjects"]),
                work["popularity"],
            ),
        )
        conn.execute(
            "INSERT INTO vibe_cards (work_id, card_json, card_text, model, prompt_version) "
            "VALUES (?, ?, ?, ?, ?)",
            (work["id"], card.model_dump_json(), card.to_text(), "fixture", VIBE_PROMPT_VERSION),
        )
        vector = work_vector(catalog["clusters"][work["cluster"]], work["id"])
        conn.execute(
            "INSERT INTO vec_vibes (work_id, embedding, media_type) VALUES (?, ?, ?)",
            (work["id"], vec_to_blob(vector), work["media_type"]),
        )
    for link in catalog["links"]:
        conn.execute(
            "INSERT INTO work_links (from_work, to_work, link_type, source, confidence, verified) "
            "VALUES (:from_work, :to_work, :link_type, :source, :confidence, :verified)",
            link,
        )
    rebuild_fts(conn)
    set_meta(conn, "embedding_model", EMBED_MODEL)
    set_meta(conn, "embedding_dims", str(EMBED_DIMS))
    set_meta(conn, "vibe_prompt_version", VIBE_PROMPT_VERSION)
    conn.commit()
    return catalog
