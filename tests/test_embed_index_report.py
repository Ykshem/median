from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from median import config, llm
from median.db import get_meta, vec_to_blob
from median.models import VibeCard
from pipeline.enrich import embed, index
from pipeline.enrich import vibe_cards as vc
from pipeline.report import build_report, check_targets, format_report
from pipeline.sample_plan import SampleSpec
from pipeline.sample_seed import upsert_work
from tests.fakes import FakeOpenAI, embedding_response
from tests.test_vibe_cards import GOOD_CARD, fetched

# embed -------------------------------------------------------------------------------------------


def _cards(db: sqlite3.Connection, n: int) -> list[int]:
    ids = []
    for i in range(1, n + 1):
        kind = "movie" if i % 2 else "book"
        work_id = upsert_work(db, fetched(f"Title {i}", kind=kind, ext=str(i)))
        vc.store_card(db, work_id, VibeCard(**GOOD_CARD), config.AGENT_MODEL)
        ids.append(work_id)
    return ids


@pytest.fixture
def openai(monkeypatch: pytest.MonkeyPatch):
    def install(*responses: Any) -> FakeOpenAI:
        fake = FakeOpenAI(list(responses))
        monkeypatch.setattr(llm, "_openai_client", lambda: fake)
        return fake

    return install


def test_embed_fills_vec_vibes_with_the_media_type_and_records_meta(
    db: sqlite3.Connection, openai
) -> None:
    ids = _cards(db, 4)
    fake = openai(embedding_response(4, prompt_tokens=400))
    report = embed.embed_cards(db)
    assert (report.embedded, report.already_done) == (4, 0)
    assert fake.calls[0]["model"] == config.EMBED_MODEL and len(fake.calls[0]["input"]) == 4
    rows = db.execute("SELECT work_id, media_type FROM vec_vibes ORDER BY work_id").fetchall()
    assert [r["work_id"] for r in rows] == ids
    assert [r["media_type"] for r in rows] == ["movie", "book", "movie", "book"]
    assert get_meta(db, "embedding_model") == config.EMBED_MODEL
    assert get_meta(db, "embedding_dims") == "1536"
    (usage,) = db.execute("SELECT * FROM llm_usage WHERE purpose = 'embed'").fetchall()
    assert usage["input_tokens"] == 400


def test_a_rerun_embeds_only_the_new_cards(db: sqlite3.Connection, openai) -> None:
    _cards(db, 3)
    fake = openai(embedding_response(3), embedding_response(1))
    embed.embed_cards(db)
    _cards(db, 4)  # one new work (Title 4); the first three are unchanged
    report = embed.embed_cards(db)
    assert (report.embedded, report.already_done) == (1, 3)
    assert len(fake.calls) == 2 and len(fake.calls[1]["input"]) == 1
    assert embed.embed_cards(db).embedded == 0 and len(fake.calls) == 2


def test_embedding_is_chunked_and_committed_per_chunk(
    db: sqlite3.Connection, openai, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embed, "CHUNK", 2)
    _cards(db, 5)
    fake = openai(embedding_response(2), embedding_response(2), embedding_response(1))
    embed.embed_cards(db)
    assert [len(c["input"]) for c in fake.calls] == [2, 2, 1]
    assert db.execute("SELECT COUNT(*) FROM vec_vibes").fetchone()[0] == 5


def test_a_new_prompt_version_discards_old_vectors_and_re_embeds_everything(
    db: sqlite3.Connection, openai
) -> None:
    _cards(db, 2)
    fake = openai(embedding_response(2), embedding_response(2))
    embed.embed_cards(db)
    db.execute("UPDATE meta SET value = 'v0' WHERE key = 'embedded_prompt_version'")
    db.commit()
    report = embed.embed_cards(db)
    assert report.embedded == 2 and report.already_done == 0 and len(fake.calls) == 2


def test_cosine_knn_works_on_the_embedded_vectors(db: sqlite3.Connection, openai) -> None:
    _cards(db, 3)
    openai(embedding_response(3))  # vector i is [i, i, ...]; all point the same way
    embed.embed_cards(db)
    query = vec_to_blob([1.0] * config.EMBED_DIMS)
    rows = db.execute(
        "SELECT work_id FROM vec_vibes WHERE embedding MATCH ? AND k = 5 AND media_type = 'book'",
        (query,),
    ).fetchall()
    assert len(rows) == 1  # only one of the three works is a book


def test_estimate_scales_with_text_length(db: sqlite3.Connection) -> None:
    _cards(db, 2)
    usd = embed.estimate_usd(embed.pending_rows(db))
    assert 0 < usd < 0.0001


# index -------------------------------------------------------------------------------------------


def test_percentile_ranks() -> None:
    assert index.percentile_ranks({1: 10.0, 2: 30.0, 3: 20.0}) == {1: 0.0, 2: 1.0, 3: 0.5}
    assert index.percentile_ranks({1: 5.0}) == {1: 1.0}
    ties = index.percentile_ranks({1: 1.0, 2: 1.0, 3: 9.0})
    assert ties[1] == ties[2] == 0.0 and ties[3] == 1.0


def test_index_ranks_within_each_media_type_and_leaves_missing_values_null(
    db: sqlite3.Connection,
) -> None:
    for ext, kind, raw in [
        ("1", "movie", 5.0),
        ("2", "movie", 50.0),
        ("3", "book", 1000.0),
        ("4", "book", 1.0),
        ("5", "book", None),
    ]:
        upsert_work(db, fetched(f"T{ext}", kind=kind, ext=ext, popularity_raw=raw))
    db.commit()
    ranked = index.build_index(db)
    assert ranked == {"movie": 2, "book": 2}
    pop = {r["title"]: r["popularity"] for r in db.execute("SELECT title, popularity FROM works")}
    assert pop == {"T1": 0.0, "T2": 1.0, "T4": 0.0, "T3": 1.0, "T5": None}
    assert index.build_index(db) == ranked  # idempotent
    assert db.execute("SELECT popularity FROM works WHERE title = 'T2'").fetchone()[0] == 1.0


def test_index_writes_meta_and_makes_titles_searchable(db: sqlite3.Connection) -> None:
    upsert_work(db, fetched("Piranesi", kind="book", ext="9"))
    db.commit()
    index.build_index(db)
    assert get_meta(db, "schema_version") == str(config.SCHEMA_VERSION)
    assert get_meta(db, "vibe_prompt_version") == config.VIBE_PROMPT_VERSION
    assert get_meta(db, "catalog_built_at").startswith("20")
    hits = db.execute("SELECT rowid FROM works_fts WHERE works_fts MATCH 'piranesi'").fetchall()
    assert len(hits) == 1


# report ------------------------------------------------------------------------------------------


def test_report_counts_the_mini_catalog(db: sqlite3.Connection, mini_catalog: dict) -> None:
    report = build_report(db)
    assert report.works == {"book": 10, "movie": 10}
    assert (report.cards, report.missing_cards, report.vectors) == (20, 0, 20)
    assert report.flagged_cards == 0
    assert report.adaptation_pairs == 5 and report.unverified_links == 1
    assert ("adaptation_of", "wikidata", 1, 5) in report.links
    assert report.spend == {}


def test_report_includes_spend_by_purpose_and_formats_readably(
    db: sqlite3.Connection, mini_catalog: dict
) -> None:
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('vibe_card', 'm', 0.1234)")
    db.execute("INSERT INTO llm_usage (purpose, model, cost_usd) VALUES ('embed', 'm', 0.0066)")
    db.commit()
    text = "\n".join(format_report(build_report(db)))
    assert "Works: 10 books, 10 movies" in text
    assert "Vibe cards: 20 / 20 (0 missing, 0 break a rule)" in text
    assert "Verified adaptation pairs: 5" in text
    assert "Spend: $0.1300 (embed $0.0066, vibe_card $0.1234)" in text


def test_targets_flag_what_is_missing(db: sqlite3.Connection, mini_catalog: dict) -> None:
    report = build_report(db)
    assert (
        check_targets(report, SampleSpec(movies=10, books=10, min_pairs=5), require_cards=True)
        == []
    )
    problems = check_targets(
        report, SampleSpec(movies=100, books=10, min_pairs=50), require_cards=True
    )
    assert any("only 10 movies" in p for p in problems)
    assert any("only 5 verified adaptation pairs" in p for p in problems)


def test_targets_notice_missing_cards_and_vectors(db: sqlite3.Connection) -> None:
    upsert_work(db, fetched("Lonely"))
    db.commit()
    problems = check_targets(
        build_report(db), SampleSpec(movies=1, books=0, min_pairs=0), require_cards=True
    )
    assert "1 works have no vibe card" in problems
    assert (
        check_targets(
            build_report(db), SampleSpec(movies=1, books=0, min_pairs=0), require_cards=False
        )
        == []
    )


def test_targets_notice_cards_without_vectors(db: sqlite3.Connection) -> None:
    work_id = upsert_work(db, fetched("Carded"))
    vc.store_card(db, work_id, VibeCard(**GOOD_CARD), config.AGENT_MODEL)
    problems = check_targets(
        build_report(db), SampleSpec(movies=1, books=0, min_pairs=0), require_cards=True
    )
    assert problems == ["0 vectors for 1 cards"]


def test_a_handful_of_works_without_cards_is_noted_not_failed(db: sqlite3.Connection) -> None:
    for i in range(200):  # 199 with cards and vectors, 1 without (0.5% missing)
        work_id = upsert_work(db, fetched(f"Work {i}", ext=str(i)))
        if i:
            vc.store_card(db, work_id, VibeCard(**GOOD_CARD), config.AGENT_MODEL)
            db.execute(
                "INSERT INTO vec_vibes (work_id, embedding, media_type) VALUES (?, ?, 'movie')",
                (work_id, vec_to_blob([0.1] * config.EMBED_DIMS)),
            )
    db.commit()
    report = build_report(db)
    assert report.missing_cards == 1 and report.missing_titles == ["Work 0 (movie)"]
    assert (
        check_targets(report, SampleSpec(movies=190, books=0, min_pairs=0), require_cards=True)
        == []
    )
    assert "Without a card: Work 0 (movie)" in "\n".join(format_report(report))
