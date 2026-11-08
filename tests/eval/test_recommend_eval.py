"""Recommendation quality on the real catalog. Run with `uv run pytest -m eval -s`.

Targets (docs/ROADMAP.md, ARCHITECTURE.md section 10): adaptation recall@10 >= 0.9, vibe recall@10
>= 0.5, and a medium-mixing share that is not near zero. Needs the built catalog with vibe vectors.
Every run writes `data/eval_report.md` listing each miss, so weights can be tuned on specifics.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime

import pytest

from median.config import (
    GOLD_ADAPTATIONS_PATH,
    GOLD_VIBES_PATH,
    REPO_ROOT,
    SCORE_WEIGHTS,
    get_settings,
)
from median.db import connect
from tests.eval import metrics

pytestmark = pytest.mark.eval

ADAPTATION_TARGET = 0.9
VIBE_TARGET = 0.5
MIXING_FLOOR = 0.15  # "not near zero"
MIXING_SAMPLE = 200
REPORT_PATH = REPO_ROOT / "data" / "eval_report.md"


@pytest.fixture(scope="module")
def conn():
    connection = connect(get_settings().db_path)
    have = connection.execute("SELECT COUNT(*) FROM vec_vibes").fetchone()[0]
    if not have:
        connection.close()
        pytest.skip("no vibe vectors yet: finish `uv run python -m pipeline.build --sample`")
    yield connection
    connection.close()


def _rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def results(conn):
    """Compute every metric once and write the report; the tests below assert on it."""
    adaptation = _rows(GOLD_ADAPTATIONS_PATH)
    pairs, in_wikidata, missing = [], [], []
    for row in adaptation:
        book = metrics.work_id(conn, "openlibrary", row["book_ol_id"])
        movie = metrics.work_id(conn, "tmdb", row["movie_tmdb_id"])
        if book is None or movie is None:
            missing.append(row["movie_title"])
            continue
        pairs.append((book, movie))
        in_wikidata.append(row["in_wikidata"] == "True")

    def subset(flag: bool) -> list[tuple[int, int]]:
        return [p for p, w in zip(pairs, in_wikidata, strict=True) if w is flag]

    vibe_pairs, vibe_missing = [], []
    for row in _rows(GOLD_VIBES_PATH):
        seed, target = (
            metrics.ref_id(conn, row["seed_ref"]),
            metrics.ref_id(conn, row["target_ref"]),
        )
        if seed is None or target is None:
            vibe_missing.append(f"{row['seed_title']} -> {row['target_title']}")
        else:
            vibe_pairs.append((seed, target))

    all_ids = [r[0] for r in conn.execute("SELECT work_id FROM vec_vibes ORDER BY work_id")]
    step = max(1, len(all_ids) // MIXING_SAMPLE)
    out = {
        "adaptation": metrics.adaptation_recall(conn, pairs),
        "adaptation_wikidata": metrics.adaptation_recall(conn, subset(True)),
        "adaptation_other": metrics.adaptation_recall(conn, subset(False)),
        "adaptation_reverse": metrics.adaptation_recall(conn, pairs, reverse=True),
        "vibe": metrics.vibe_recall(conn, vibe_pairs),
        "mixing": metrics.medium_mixing(conn, all_ids[::step]),
        "missing": missing,
        "vibe_missing": vibe_missing,
    }
    _write_report(out)
    return out


def _line(name: str, recall: metrics.Recall, target: float | None = None) -> str:
    goal = f" (target >= {target})" if target else ""
    return f"- {name}: {recall.hits}/{recall.total} = {recall.value:.3f}{goal}"


def _write_report(r: dict) -> None:
    lines = [
        "# Recommendation eval",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · weights {SCORE_WEIGHTS}",
        "",
        _line("Adaptation recall@10, book -> film", r["adaptation"], ADAPTATION_TARGET),
        _line("  of which in Wikidata", r["adaptation_wikidata"]),
        _line("  of which NOT in Wikidata", r["adaptation_other"]),
        _line("Adaptation recall@10, film -> book (extra)", r["adaptation_reverse"]),
        _line("Vibe recall@10 (proposed gold pairs)", r["vibe"], VIBE_TARGET),
        f"- Medium mixing, 'any' results in the other medium: {r['mixing']:.3f} "
        f"(floor {MIXING_FLOOR})",
        "",
    ]
    for title, recall in [
        ("Adaptation misses (book -> film)", r["adaptation"]),
        ("Adaptation misses (film -> book)", r["adaptation_reverse"]),
        ("Vibe misses", r["vibe"]),
    ]:
        lines += [f"## {title}", ""] + [f"- {m}" for m in recall.misses or ["(none)"]] + [""]
    if r["missing"] or r["vibe_missing"]:
        lines += ["## Gold entries not in the catalog", ""]
        lines += [f"- {m}" for m in [*r["missing"], *r["vibe_missing"]]] + [""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def test_every_gold_entry_is_in_the_catalog(results) -> None:
    assert results["missing"] == [] and results["vibe_missing"] == []


def test_adaptation_recall_at_10(results, capsys) -> None:
    with capsys.disabled():
        print(f"\n{_line('adaptation recall@10', results['adaptation'], ADAPTATION_TARGET)}")
        print(_line("  in Wikidata", results["adaptation_wikidata"]))
        print(_line("  not in Wikidata", results["adaptation_other"]))
        print(_line("  film -> book", results["adaptation_reverse"]))
        print(f"  misses: {results['adaptation'].misses}")
    assert results["adaptation"].value >= ADAPTATION_TARGET, results["adaptation"].misses


def test_vibe_recall_at_10(results, capsys) -> None:
    with capsys.disabled():
        print(f"\n{_line('vibe recall@10', results['vibe'], VIBE_TARGET)}")
        print(f"  misses: {results['vibe'].misses}")
    assert results["vibe"].value >= VIBE_TARGET, results["vibe"].misses


def test_the_vibe_cards_do_not_leak_the_medium(results, capsys) -> None:
    with capsys.disabled():
        print(f"\nmedium mixing: {results['mixing']:.3f} (floor {MIXING_FLOOR})")
    assert results["mixing"] > MIXING_FLOOR
