"""Finish the catalog: search index, popularity percentiles, metadata, a shareable single file."""

from __future__ import annotations

import logging
import sqlite3
from bisect import bisect_left
from datetime import UTC, datetime

from median.config import SCHEMA_VERSION, VIBE_PROMPT_VERSION
from median.db import rebuild_fts, set_meta

logger = logging.getLogger(__name__)


def percentile_ranks(values: dict[int, float]) -> dict[int, float]:
    """Map each id to its percent rank in 0..1: the share of the other values that are lower.

    Equal values get equal ranks. A single value gets 1.0.
    """
    if len(values) == 1:
        return {key: 1.0 for key in values}
    ordered = sorted(values.values())
    span = len(ordered) - 1
    return {key: bisect_left(ordered, value) / span for key, value in values.items()}


def build_index(conn: sqlite3.Connection) -> dict[str, int]:
    """Rebuild FTS, set `popularity` per media type, write `meta`, and checkpoint the WAL.

    Idempotent. Returns how many works got a popularity value, by media type.
    """
    rebuild_fts(conn)
    ranked: dict[str, int] = {}
    media_types = [r[0] for r in conn.execute("SELECT DISTINCT media_type FROM works")]
    for media in media_types:
        raw = {
            r["id"]: r["popularity_raw"]
            for r in conn.execute(
                "SELECT id, popularity_raw FROM works "
                "WHERE media_type = ? AND popularity_raw IS NOT NULL",
                (media,),
            )
        }
        if raw:
            conn.executemany(
                "UPDATE works SET popularity = ? WHERE id = ?",
                [(pct, work_id) for work_id, pct in percentile_ranks(raw).items()],
            )
        ranked[media] = len(raw)
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    set_meta(conn, "vibe_prompt_version", VIBE_PROMPT_VERSION)
    set_meta(conn, "catalog_built_at", datetime.now(UTC).isoformat(timespec="seconds"))
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    logger.info("index: popularity set for %s", ranked)
    return ranked
