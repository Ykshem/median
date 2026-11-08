from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from median import config, llm
from pipeline.enrich import link_check, match
from pipeline.sources import tmdb
from tests.fakes import FakeBatchClient
from tests.fixtures.source_payloads import tmdb_details


def _movie_payload(movie_id: int, title: str, date: str, *crew: tuple[str, str]) -> dict:
    payload = tmdb_details(movie_id, title, date)
    payload["credits"] = {"crew": [{"name": n, "job": j} for n, j in crew]}
    return payload


class World:
    """A tiny catalog plus a fake TMDB that serves each movie's credits."""

    def __init__(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        self.db = db
        self.payloads: dict[int, dict] = {}
        self.client = tmdb.make_client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json=self.payloads[int(r.url.path.rsplit("/", 1)[-1])]
                )
            ),
            cache_dir=tmp_path / "tmdb",
            sleep=lambda s: None,
        )

    def movie(self, wid: int, tmdb_id: int, title: str, year: int, *crew: tuple[str, str]) -> None:
        self.db.execute(
            "INSERT INTO works (id, media_type, title, year, synopsis) "
            "VALUES (?, 'movie', ?, ?, ?)",
            (wid, title, year, f"Film synopsis of {title}"),
        )
        self.db.execute("INSERT INTO external_ids VALUES (?, 'tmdb', ?)", (wid, str(tmdb_id)))
        self.payloads[tmdb_id] = _movie_payload(tmdb_id, title, f"{year}-01-01", *crew)

    def book(self, wid: int, ol_id: str, title: str, year: int | None, *authors: str) -> None:
        self.db.execute(
            "INSERT INTO works (id, media_type, title, year, creators, synopsis) "
            "VALUES (?, 'book', ?, ?, ?, ?)",
            (wid, title, year, json.dumps(list(authors)), f"Book synopsis of {title}"),
        )
        self.db.execute("INSERT INTO external_ids VALUES (?, 'openlibrary', ?)", (wid, ol_id))


@pytest.fixture
def world(db: sqlite3.Connection, tmp_path: Path):
    w = World(db, tmp_path)
    yield w
    w.client.close()


def _cands(world: World) -> dict[tuple[int, int], match.Candidate]:
    return {(c.movie_id, c.book_id): c for c in match.find_candidates(world.db, world.client)}


# match -------------------------------------------------------------------------------------------


def test_novel_credit_plus_similar_title_is_a_confident_candidate(world: World) -> None:
    world.movie(1, 100, "The Shining", 1980, ("Stephen King", "Novel"))
    world.book(2, "OL1W", "The Shining", 1977, "Stephen King")
    cand = _cands(world)[(1, 2)]
    assert cand.source == "tmdb_credits" and cand.confidence == pytest.approx(0.9)


def test_the_only_catalog_book_by_the_credited_author_is_a_weaker_candidate(world: World) -> None:
    world.movie(1, 100, "Different Name", 2000, ("J. R. R. Tolkien", "Novel"))
    world.book(2, "OL1W", "The Hobbit", 1937, "J.R.R. Tolkien")
    cand = _cands(world)[(1, 2)]
    assert cand.source == "tmdb_credits" and cand.confidence == 0.6  # name spacing is ignored


def test_an_author_with_several_books_needs_a_title_match(world: World) -> None:
    world.movie(1, 100, "Misery", 1990, ("Stephen King", "Novel"))
    world.book(2, "OL1W", "Carrie", 1974, "Stephen King")
    world.book(3, "OL2W", "Misery", 1987, "Stephen King")
    cands = _cands(world)
    assert (1, 3) in cands and (1, 2) not in cands


def test_a_book_published_after_the_film_is_not_its_source(world: World) -> None:
    world.movie(1, 100, "Old Film", 1980, ("Some Author", "Novel"))
    world.book(2, "OL1W", "Old Film", 1995, "Some Author")
    assert _cands(world) == {}


def test_other_credit_jobs_do_not_count_as_a_novel_credit(world: World) -> None:
    world.movie(1, 100, "Heist", 2000, ("Some Author", "Screenplay"))
    world.book(2, "OL1W", "Heist", 1990, "Some Author")
    cand = _cands(world)[(1, 2)]
    assert cand.source == "title_match"  # only the equal title connects them


def test_equal_title_without_a_credit_is_a_low_confidence_candidate(world: World) -> None:
    world.movie(1, 100, "Room", 2015)
    world.book(2, "OL1W", "Room", 2010, "Emma Donoghue")
    world.book(3, "OL2W", "Room", None, "Unknown Year")  # needs a known year to be trusted
    cands = _cands(world)
    assert cands[(1, 2)].source == "title_match" and cands[(1, 2)].confidence == 0.4
    assert (1, 3) not in cands


def test_existing_links_are_not_proposed_again(world: World) -> None:
    world.movie(1, 100, "The Shining", 1980, ("Stephen King", "Novel"))
    world.book(2, "OL1W", "The Shining", 1977, "Stephen King")
    world.db.execute("INSERT INTO work_links VALUES (1, 2, 'adaptation_of', 'wikidata', 1.0, 1)")
    assert _cands(world) == {}


def test_match_links_inserts_unverified_candidates_once(world: World) -> None:
    world.movie(1, 100, "The Shining", 1980, ("Stephen King", "Novel"))
    world.book(2, "OL1W", "The Shining", 1977, "Stephen King")
    assert match.match_links(world.db, world.client) == {"tmdb_credits": 1, "title_match": 0}
    assert match.match_links(world.db, world.client) == {"tmdb_credits": 0, "title_match": 0}
    (row,) = world.db.execute("SELECT * FROM work_links").fetchall()
    assert (row["link_type"], row["verified"], row["source"]) == (
        "adaptation_of",
        0,
        "tmdb_credits",
    )


# link_check --------------------------------------------------------------------------------------


def _verdict(adaptation: bool, confidence: float) -> str:
    return json.dumps({"is_adaptation": adaptation, "confidence": confidence, "reason": "r"})


@pytest.fixture
def batch(monkeypatch: pytest.MonkeyPatch):
    def install(answer) -> FakeBatchClient:
        fake = FakeBatchClient(answer)
        monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
        return fake

    return install


def _three_links(world: World) -> None:
    for i, title in enumerate(["Alpha", "Beta", "Gamma"], start=1):
        world.movie(i, 100 + i, title, 2000, ("A Writer", "Screenplay"))
        world.book(10 + i, f"OL{i}W", title, 1990, f"Author {i}")
        world.db.execute(
            "INSERT INTO work_links VALUES (?, ?, 'adaptation_of', 'title_match', 0.4, 0)",
            (i, 10 + i),
        )
    world.db.commit()


def _check(world: World, tmp_path: Path):
    return link_check.check_links(
        world.db, world.client, cache_path=tmp_path / "verdicts.json", sleep=lambda s: None
    )


def test_parse_verdict_tolerates_fences_and_rejects_bad_confidence() -> None:
    assert link_check.parse_verdict("```json\n" + _verdict(True, 0.9) + "\n```").keep
    assert not link_check.parse_verdict(_verdict(True, 0.69)).keep
    assert not link_check.parse_verdict(_verdict(False, 0.99)).keep
    with pytest.raises(ValueError):
        link_check.parse_verdict(_verdict(True, 1.5))


def test_confident_yes_is_kept_everything_else_is_dropped(
    world: World, batch, tmp_path: Path
) -> None:
    _three_links(world)
    answers = {
        "l1_11": _verdict(True, 0.95),
        "l2_12": _verdict(True, 0.5),
        "l3_13": _verdict(False, 0.9),
    }
    batch(lambda r: answers[r["custom_id"]])
    report = _check(world, tmp_path)
    assert (report.kept, report.dropped, report.unknown) == (1, 2, [])
    (row,) = world.db.execute("SELECT * FROM work_links").fetchall()
    assert (row["from_work"], row["verified"], row["confidence"]) == (1, 1, 0.95)


def test_the_prompt_carries_both_works_and_the_film_writers(
    world: World, batch, tmp_path: Path
) -> None:
    _three_links(world)
    fake = batch(lambda r: _verdict(True, 0.9))
    _check(world, tmp_path)
    prompt = fake.created[0][0]["params"]["messages"][0]["content"]
    assert "film: Alpha (2000); writers: A Writer" in prompt
    assert "book: Alpha (1990); author: Author 1" in prompt
    assert "<work_data>" in prompt and "not instructions" in prompt


def test_an_unusable_reply_leaves_the_link_for_a_later_run(
    world: World, batch, tmp_path: Path
) -> None:
    _three_links(world)
    batch(lambda r: "no idea" if r["custom_id"] == "l2_12" else _verdict(True, 0.9))
    report = _check(world, tmp_path)
    assert report.kept == 2 and len(report.unknown) == 1
    assert world.db.execute("SELECT COUNT(*) FROM work_links WHERE verified = 0").fetchone()[0] == 1


def test_a_rejected_link_proposed_again_is_not_paid_for_twice(
    world: World, batch, tmp_path: Path
) -> None:
    _three_links(world)
    fake = batch(lambda r: _verdict(False, 0.9))
    _check(world, tmp_path)
    assert world.db.execute("SELECT COUNT(*) FROM work_links").fetchone()[0] == 0
    world.db.execute(
        "INSERT INTO work_links VALUES (1, 11, 'adaptation_of', 'title_match', 0.4, 0)"
    )
    report = _check(world, tmp_path)  # `match` proposed it again
    assert report.dropped == 1 and len(fake.created) == 1  # judged from the cache, no new batch
    assert world.db.execute("SELECT COUNT(*) FROM work_links").fetchone()[0] == 0


def test_link_check_logs_batch_priced_usage_and_estimates_before_spending(
    world: World, batch, tmp_path: Path
) -> None:
    _three_links(world)
    cache = tmp_path / "verdicts.json"
    count, cost = link_check.estimate_link_check(world.db, world.client, cache)
    assert count == 3 and 0 < cost < 0.01
    batch(lambda r: (_verdict(True, 0.9), 1_000_000, 1_000_000))
    _check(world, tmp_path)
    (row,) = world.db.execute("SELECT * FROM llm_usage WHERE purpose = 'link_check'").fetchall()
    assert row["cost_usd"] == pytest.approx(0.30 * 3)  # 3 requests at half of $0.60 per million
    assert link_check.estimate_link_check(world.db, world.client, cache)[0] == 0


def test_nothing_pending_makes_no_batch(world: World, batch, tmp_path: Path) -> None:
    fake = batch(lambda r: _verdict(True, 0.9))
    report = _check(world, tmp_path)
    assert (report.kept, report.dropped) == (0, 0) and fake.created == []
    assert config.LINK_PROMPT_VERSION == "v1"


# tightened matching (found by inspecting real candidates) -----------------------------------------


def test_title_score_does_not_rate_unrelated_titles_as_similar() -> None:
    assert match.title_score("the cat in the hat", "fox in socks") < match.TITLE_SCORE
    assert match.title_score("the hobbit an unexpected journey", "the lord of the rings") < 80
    assert match.title_score("fifty shades freed", "fifty shades darker") < match.TITLE_SCORE
    assert match.title_score("the da vinci code", "the da vinci code") == 100
    contained = match.title_score("the hunger games mockingjay part 1", "the hunger games")
    assert contained == match.CONTAINED_TITLE_SCORE
    assert match.title_score("harry potter", "harry potter and the stone") >= match.TITLE_SCORE


def test_a_similar_but_different_author_name_is_not_a_match(world: World) -> None:
    world.movie(1, 100, "The Wild Robot", 2024, ("Peter Brown", "Novel"))
    world.book(
        2, "OL1W", "Make It Stick", None, "Peter C. Brown"
    )  # a different, non-fiction author
    assert _cands(world) == {}


def test_reordered_and_punctuated_names_still_match(world: World) -> None:
    world.movie(1, 100, "Heat", 1995, ("Tolkien, J.R.R.", "Novel"))
    world.book(2, "OL1W", "Anything", 1950, "J.R.R. Tolkien")
    assert (1, 2) in _cands(world)


def test_a_film_subtitle_still_finds_its_book(world: World) -> None:
    world.movie(1, 100, "The Hunger Games: Mockingjay - Part 1", 2014, ("Suzanne Collins", "Novel"))
    world.book(2, "OL1W", "The Hunger Games", 2008, "Suzanne Collins")
    world.book(3, "OL2W", "Gregor the Overlander", 2003, "Suzanne Collins")
    cands = _cands(world)
    assert (1, 2) in cands and (1, 3) not in cands
    assert cands[(1, 2)].confidence == pytest.approx(0.87)
