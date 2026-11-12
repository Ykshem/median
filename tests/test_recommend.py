from __future__ import annotations

import json
import sqlite3

import pytest

from median import config, recommend, taste
from median.db import vec_to_blob
from median.models import VibeCard
from tests.test_taste import add_work, shelve, vec

W = config.SCORE_WEIGHTS


def link(db, from_work, to_work, link_type="adaptation_of", confidence=1.0, verified=1) -> None:
    db.execute(
        "INSERT INTO work_links (from_work, to_work, link_type, source, confidence, verified) "
        "VALUES (?, ?, ?, 'wikidata', ?, ?)",
        (from_work, to_work, link_type, confidence, verified),
    )


def card(db, work_id: int, themes: list[str], mood: list[str]) -> None:
    # A real card needs >= 3 themes and moods; pad with terms unique to this work so the padding
    # can never count as something two works share.
    themes = themes + [f"filler-theme-{work_id}-{i}" for i in range(3 - len(themes))]
    mood = mood + [f"filler-mood-{work_id}-{i}" for i in range(3 - len(mood))]
    c = VibeCard(
        logline="x",
        themes=themes,
        mood=mood,
        tone="t",
        setting="s",
        pacing="slow",
        intensity="low",
        audience="adult",
        hooks=["h1", "h2"],
    )
    db.execute(
        "INSERT INTO vibe_cards (work_id, card_json, card_text, model, prompt_version) "
        "VALUES (?, ?, ?, 'm', 'v1')",
        (work_id, c.model_dump_json(), c.to_text()),
    )


def titles(recs) -> list[int]:
    return [r.work_id for r in recs]


@pytest.fixture
def world(db: sqlite3.Connection) -> sqlite3.Connection:
    """Seed film 1 (vector a). Book 2 is its verified source but its vector is unrelated; book 3 is
    close in vibe but unlinked; book 4 is far; film 5 is close in vibe."""
    add_work(db, 1, vec(a=1), "movie")
    add_work(db, 2, vec(e=1), "book")
    add_work(db, 3, vec(a=0.95, b=0.3), "book")
    add_work(db, 4, vec(g=1), "book")
    add_work(db, 5, vec(a=0.9, c=0.4), "movie")
    db.execute("UPDATE works SET title = 'The Film' WHERE id = 1")
    db.execute("UPDATE works SET title = 'The Source Book' WHERE id = 2")
    link(db, 1, 2)  # film 1 is adapted from book 2
    db.commit()
    return db


# links and vibes ---------------------------------------------------------------------------------


def test_a_verified_link_outranks_a_closer_vibe(world: sqlite3.Connection) -> None:
    recs = recommend.overlap(world, 1, "book")
    assert titles(recs)[:2] == [2, 3]  # the linked book beats the near-identical vibe
    assert recs[0].link == 1.0 and recs[1].link == 0.0


def test_the_score_is_the_weighted_sum_of_its_parts(world: sqlite3.Connection) -> None:
    world.execute("UPDATE works SET popularity = 0.5 WHERE id = 2")
    rec = recommend.overlap(world, 1, "book")[0]
    expected = W["link"] * 1.0 + W["vibe"] * rec.vibe + W["popularity"] * 0.5
    assert rec.score == pytest.approx(expected, abs=1e-5)
    assert rec.popularity == 0.5 and rec.taste == 0.0


def test_a_linked_work_with_an_unrelated_vector_still_gets_a_vibe_value(
    world: sqlite3.Connection,
) -> None:
    rec = recommend.overlap(world, 1, "book")[0]
    assert rec.work_id == 2 and rec.vibe == pytest.approx(0.0, abs=1e-6)  # orthogonal, not missing


def test_the_target_medium_filters_both_links_and_neighbours(world: sqlite3.Connection) -> None:
    assert set(titles(recommend.overlap(world, 1, "book"))) == {2, 3, 4}
    assert titles(recommend.overlap(world, 1, "movie")) == [5]
    assert set(titles(recommend.overlap(world, 1, "any"))) == {2, 3, 4, 5}


def test_the_seed_is_never_recommended(world: sqlite3.Connection) -> None:
    for target in ("book", "movie", "any"):
        assert 1 not in titles(recommend.overlap(world, 1, target))


def test_links_work_in_both_directions(world: sqlite3.Connection) -> None:
    recs = recommend.overlap(world, 2, "movie")
    assert recs[0].work_id == 1 and recs[0].link == 1.0


def test_unverified_links_do_not_count(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    add_work(db, 2, vec(e=1), "book")
    link(db, 1, 2, verified=0)
    assert recommend.overlap(db, 1, "book")[0].link == 0.0


def test_link_confidence_scales_the_strength(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    add_work(db, 2, vec(e=1), "book")
    link(db, 1, 2, confidence=0.8)
    assert recommend.overlap(db, 1, "book")[0].link == pytest.approx(0.8)


def test_link_types_have_different_strengths(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    for i, kind in enumerate(["adaptation_of", "same_franchise", "same_creator"], start=2):
        add_work(db, i, vec(e=1), "book")
        link(db, 1, i, kind)
    strengths = {r.work_id: r.link for r in recommend.overlap(db, 1, "book")}
    assert strengths == {2: 1.0, 3: 0.8, 4: 0.5}


def test_the_strongest_link_wins_when_two_connect_the_same_pair(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    add_work(db, 2, vec(e=1), "book")
    link(db, 1, 2, "same_creator")
    link(db, 1, 2, "adaptation_of")
    assert recommend.overlap(db, 1, "book")[0].link == 1.0


def test_a_seed_without_a_vector_falls_back_to_links_only(db: sqlite3.Connection) -> None:
    add_work(db, 1, None, "movie")
    add_work(db, 2, vec(a=1), "book")
    add_work(db, 3, vec(a=1), "book")
    link(db, 1, 2)
    recs = recommend.overlap(db, 1, "book")
    assert titles(recs) == [2] and recs[0].vibe == 0.0


def test_results_are_deterministic_and_ties_break_by_id(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    for i in (5, 3, 4):
        add_work(db, i, vec(a=1), "book")
    assert titles(recommend.overlap(db, 1, "book")) == [3, 4, 5]


def test_k_limits_the_results(world: sqlite3.Connection) -> None:
    assert len(recommend.overlap(world, 1, "any", k=2)) == 2


def test_bad_arguments_are_rejected(world: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="target_media"):
        recommend.overlap(world, 1, "podcast")
    with pytest.raises(ValueError, match="at least 1"):
        recommend.overlap(world, 1, "book", k=0)
    with pytest.raises(ValueError, match="unknown work"):
        recommend.overlap(world, 999, "book")
    with pytest.raises(ValueError, match="unknown persona"):
        recommend.overlap(world, 1, "book", persona_id=999)


# franchise ---------------------------------------------------------------------------------------


def test_at_most_two_results_come_from_one_franchise(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    for i in (2, 3, 4):  # a three-book series, all strongly linked to the seed
        add_work(db, i, vec(a=1), "book")
        link(db, 1, i, "same_franchise")
    add_work(db, 5, vec(a=0.5, b=0.8), "book")  # unrelated, weaker
    link(db, 2, 3, "same_franchise")
    link(db, 3, 4, "same_franchise")
    recs = recommend.overlap(db, 1, "book")
    series = [r.work_id for r in recs if r.work_id in (2, 3, 4)]
    assert len(series) == config.MAX_PER_FRANCHISE == 2
    assert 5 in titles(recs)  # the slot freed by the cap goes to the next best


def test_a_series_neighbours_links_count_at_a_reduced_strength(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    add_work(db, 2, vec(e=1), "book")  # the series neighbour
    add_work(db, 3, vec(f=1), "movie")  # an adaptation of the neighbour
    link(db, 1, 2, "same_franchise")
    link(db, 3, 2, "adaptation_of")
    recs = {r.work_id: r for r in recommend.overlap(db, 1, "movie")}
    assert recs[3].link == pytest.approx(config.LINK_STRENGTH["same_franchise"])
    assert any("same series" in reason for reason in recs[3].reasons)


# shelves and taste -------------------------------------------------------------------------------


def test_works_on_the_persona_shelves_are_left_out(world: sqlite3.Connection) -> None:
    shelve(world, 1, 2, rating=9)
    recs = recommend.overlap(world, 1, "book", persona_id=1)
    assert 2 not in titles(recs) and 3 in titles(recs)
    assert 2 in titles(recommend.overlap(world, 1, "book"))  # no persona, no filter


def test_the_taste_vector_breaks_a_tie_between_equal_vibes(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    add_work(db, 2, vec(a=0.6, b=0.8), "book")  # equally far from the seed,
    add_work(db, 3, vec(a=0.6, c=0.8), "book")  # in different directions
    db.execute("UPDATE personas SET taste_vector = ? WHERE id = 1", (vec_to_blob(vec(c=1)),))
    assert titles(recommend.overlap(db, 1, "book"))[0] == 2  # tie goes to the lower id
    with_taste = recommend.overlap(db, 1, "book", persona_id=1)
    assert titles(with_taste)[0] == 3
    assert with_taste[0].taste == pytest.approx(0.8)
    assert with_taste[0].score - with_taste[1].score == pytest.approx(W["taste"] * 0.8)


def test_a_persona_without_a_taste_vector_adds_nothing(world: sqlite3.Connection) -> None:
    plain = [(r.work_id, r.score) for r in recommend.overlap(world, 1, "book")]
    with_persona = [(r.work_id, r.score) for r in recommend.overlap(world, 1, "book", persona_id=1)]
    assert plain == with_persona


# reasons -----------------------------------------------------------------------------------------


def test_reasons_say_how_a_film_and_its_source_book_are_connected(
    world: sqlite3.Connection,
) -> None:
    assert "Source of The Film" in recommend.overlap(world, 1, "book")[0].reasons
    recs = recommend.overlap(world, 2, "movie")
    assert "Adapted from The Source Book" in recs[0].reasons


def test_shared_themes_and_moods_are_listed_up_to_the_limit(world: sqlite3.Connection) -> None:
    card(world, 1, ["grief", "family", "ambition", "loss"], ["eerie", "warm", "tense"])
    card(world, 3, ["grief", "ambition", "loss", "heist"], ["eerie", "cozy", "tense"])
    reasons = {r.work_id: r.reasons for r in recommend.overlap(world, 1, "book")}
    assert reasons[3] == ["Shared themes: grief, ambition, loss"]  # capped at MAX_REASON_THEMES


def test_a_vibe_only_result_with_nothing_shared_says_similar_vibe(
    world: sqlite3.Connection,
) -> None:
    assert recommend.overlap(world, 1, "movie")[0].reasons == ["Similar vibe"]


def test_a_taste_reason_appears_only_when_the_taste_match_is_strong(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "movie")
    add_work(db, 2, vec(c=1), "book")
    add_work(db, 3, vec(d=1), "book")
    db.execute("UPDATE personas SET taste_vector = ? WHERE id = 1", (vec_to_blob(vec(c=1)),))
    reasons = {r.work_id: r.reasons for r in recommend.overlap(db, 1, "book", persona_id=1)}
    assert "Fits Horror fan's taste" in reasons[2]
    assert not any("taste" in text for text in reasons.get(3, []))


# for_persona -------------------------------------------------------------------------------------


@pytest.fixture
def tastes(db: sqlite3.Connection) -> sqlite3.Connection:
    """Persona 1 loves works near c; candidates are near c, near d, and one popular outlier."""
    add_work(db, 1, vec(c=1), "movie")
    add_work(db, 2, vec(c=1, b=0.1), "movie")
    add_work(db, 3, vec(c=0.9, b=0.2), "book")
    add_work(db, 4, vec(d=1), "book")
    add_work(db, 5, vec(f=1), "book")
    add_work(db, 6, vec(c=0.8, f=0.5), "book")  # near the taste, and leaning toward f
    db.execute("UPDATE works SET popularity = 0.2 WHERE id = 3")
    db.execute("UPDATE works SET popularity = 1.0 WHERE id = 5")
    card(db, 1, ["dread", "family"], ["eerie"])
    card(db, 3, ["dread", "grief"], ["eerie", "tense"])
    for work_id in (1, 2):
        shelve(db, 1, work_id, rating=10)
    shelve(db, 1, 4, rating=1)
    taste.recompute(db, 1)
    db.commit()
    return db


def test_for_persona_returns_works_near_the_taste_vector(tastes: sqlite3.Connection) -> None:
    recs = recommend.for_persona(tastes, 1, "book")
    assert titles(recs)[0] == 3  # near the taste; work 4 is on the shelf (rated 1), so it's gone
    assert 4 not in titles(recs)
    assert recs[0].link == 0.0 and recs[0].taste > 0.8


def test_for_persona_leaves_out_shelved_works_and_respects_the_medium(
    tastes: sqlite3.Connection,
) -> None:
    for rec in recommend.for_persona(tastes, 1, "movie"):
        assert rec.work_id not in (1, 2)
    assert set(titles(recommend.for_persona(tastes, 1, "book"))) <= {3, 5, 6}


def test_for_persona_explains_with_the_personas_liked_themes(tastes: sqlite3.Connection) -> None:
    reasons = recommend.for_persona(tastes, 1, "book")[0].reasons
    assert "Shared themes: dread, eerie" in reasons
    assert "Fits Horror fan's taste" in reasons


def test_a_mood_vector_blends_with_the_taste_vector(tastes: sqlite3.Connection) -> None:
    base = recommend.for_persona(tastes, 1, "book")
    steered = recommend.for_persona(tastes, 1, "book", mood_vector=vec(f=1))
    assert titles(base)[0] == 3
    assert steered[0].work_id == 6  # near the taste AND toward the mood: it overtakes work 3
    assert "Matches your mood request" in steered[0].reasons


def test_a_higher_mood_weight_lets_the_request_win_over_the_taste(
    tastes: sqlite3.Connection,
) -> None:
    equal = recommend.for_persona(tastes, 1, "book", mood_vector=vec(f=1))
    assert recommend.for_persona(tastes, 1, "book", mood_vector=vec(f=1), mood_weight=1.0) == equal
    strong = recommend.for_persona(tastes, 1, "book", mood_vector=vec(f=1), mood_weight=20.0)
    assert titles(equal)[0] == 6  # the blend favours the work near both
    assert titles(strong)[0] == 5  # the pure mood match, far from the taste, now wins


def test_a_mood_vector_alone_works_without_a_taste_vector(db: sqlite3.Connection) -> None:
    add_work(db, 1, vec(a=1), "book")
    add_work(db, 2, vec(b=1), "book")
    recs = recommend.for_persona(db, 1, "book", mood_vector=vec(b=1))
    assert titles(recs)[0] == 2 and recs[0].taste == 0.0


def test_with_no_taste_and_no_mood_the_most_popular_unshelved_work_wins(
    db: sqlite3.Connection,
) -> None:
    for i, pop in [(1, 0.1), (2, 0.9), (3, 0.5)]:
        add_work(db, i, vec(a=1), "book")
        db.execute("UPDATE works SET popularity = ? WHERE id = ?", (pop, i))
    shelve(db, 1, 2, rating=8)
    recs = recommend.for_persona(db, 1, "book")
    assert titles(recs) == [3, 1] and recs[0].reasons == ["Popular pick"]


def test_for_persona_rejects_bad_arguments(tastes: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="unknown persona"):
        recommend.for_persona(tastes, 999)
    with pytest.raises(ValueError, match="target_media"):
        recommend.for_persona(tastes, 1, "podcast")


def test_embed_query_logs_usage_through_the_wrapper(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from median import llm
    from tests.fakes import FakeOpenAI, embedding_response

    fake = FakeOpenAI([embedding_response(1, prompt_tokens=7)])
    monkeypatch.setattr(llm, "_openai_client", lambda: fake)
    vector = recommend.embed_query(db, "something cozy and sad")
    assert len(vector) == config.EMBED_DIMS and fake.calls[0]["input"] == ["something cozy and sad"]
    (row,) = db.execute("SELECT purpose, input_tokens FROM llm_usage").fetchall()
    assert (row["purpose"], row["input_tokens"]) == ("mood_query", 7)


def test_the_recommendation_json_shape_is_stable(world: sqlite3.Connection) -> None:
    rec = recommend.overlap(world, 1, "book")[0]
    assert set(json.loads(json.dumps(rec.__dict__))) == {
        "work_id",
        "score",
        "reasons",
        "link",
        "vibe",
        "taste",
        "popularity",
    }
