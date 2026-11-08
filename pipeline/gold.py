"""Gold evaluation files: draft them from titles, read them back, keep ids stable.

`tests/eval/gold_adaptations.csv` lists known film/book adaptation pairs (the M2 eval checks that
`overlap(book, 'movie')` finds the film). `tests/eval/gold_vibes.csv` lists cross-media pairs that
"feel alike" (the vibe recall eval). Both are keyed by TMDB id / Open Library work id, not by
database ids, so they survive rebuilds.

Drafting resolves titles through the same TMDB / Open Library lookups as the review set and
accepts exact matches only: a wrong id here would silently corrupt the eval.

    uv run python -m pipeline.gold --draft-adaptations
    uv run python -m pipeline.gold --draft-vibes
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from median.config import GOLD_ADAPTATIONS_PATH, GOLD_VIBES_PATH
from pipeline.sources import SkipWork, openlibrary, tmdb, wikidata
from pipeline.sources.http import CachedClient

logger = logging.getLogger(__name__)

ADAPTATION_FIELDS = [
    "book_ol_id",
    "movie_tmdb_id",
    "book_title",
    "book_author",
    "movie_title",
    "movie_year",
    "in_wikidata",
]
VIBE_FIELDS = ["seed_ref", "target_ref", "seed_title", "target_title", "why", "status"]


@dataclass(frozen=True)
class AdaptationSeed:
    movie_title: str
    movie_year: int
    book_title: str
    book_author: str


@dataclass(frozen=True)
class VibeSeed:
    """One 'feels alike' pair, as (kind, title, year-or-author) for the seed and target."""

    seed: tuple[str, str, str | int]  # ("movie", title, year) or ("book", title, author)
    target: tuple[str, str, str | int]
    why: str


def A(movie: str, year: int, book: str, author: str) -> AdaptationSeed:  # noqa: N802
    return AdaptationSeed(movie, year, book, author)


# Well-known adaptations; some are missing from Wikidata on purpose, so the M2 eval can report
# "Wikidata-covered vs not". More than 50 are listed because a few won't resolve exactly.
ADAPTATION_SEEDS = [
    A("The Shining", 1980, "The Shining", "Stephen King"),
    A("Gone Girl", 2014, "Gone Girl", "Gillian Flynn"),
    A(
        "The Lord of the Rings: The Fellowship of the Ring",
        2001,
        "The Fellowship of the Ring",
        "J.R.R. Tolkien",
    ),
    A("Pride & Prejudice", 2005, "Pride and Prejudice", "Jane Austen"),
    A("The Martian", 2015, "The Martian", "Andy Weir"),
    A("Jaws", 1975, "Jaws", "Peter Benchley"),
    A("The Godfather", 1972, "The Godfather", "Mario Puzo"),
    A("Jurassic Park", 1993, "Jurassic Park", "Michael Crichton"),
    A("The Silence of the Lambs", 1991, "The Silence of the Lambs", "Thomas Harris"),
    A("Fight Club", 1999, "Fight Club", "Chuck Palahniuk"),
    A("The Hunger Games", 2012, "The Hunger Games", "Suzanne Collins"),
    A(
        "Harry Potter and the Philosopher's Stone",
        2001,
        "Harry Potter and the Philosopher's Stone",
        "J. K. Rowling",
    ),
    A("The Hobbit: An Unexpected Journey", 2012, "The Hobbit", "J.R.R. Tolkien"),
    A("Dune", 2021, "Dune", "Frank Herbert"),
    A("The Great Gatsby", 2013, "The Great Gatsby", "F. Scott Fitzgerald"),
    A("To Kill a Mockingbird", 1962, "To Kill a Mockingbird", "Harper Lee"),
    A("One Flew Over the Cuckoo's Nest", 1975, "One Flew Over the Cuckoo's Nest", "Ken Kesey"),
    A("The Handmaid's Tale", 1990, "The Handmaid's Tale", "Margaret Atwood"),
    A("Life of Pi", 2012, "Life of Pi", "Yann Martel"),
    A("The Girl with the Dragon Tattoo", 2011, "The Girl with the Dragon Tattoo", "Stieg Larsson"),
    A("Little Women", 2019, "Little Women", "Louisa May Alcott"),
    A("Emma.", 2020, "Emma", "Jane Austen"),
    A("Atonement", 2007, "Atonement", "Ian McEwan"),
    A("The Road", 2009, "The Road", "Cormac McCarthy"),
    A("No Country for Old Men", 2007, "No Country for Old Men", "Cormac McCarthy"),
    A("Never Let Me Go", 2010, "Never Let Me Go", "Kazuo Ishiguro"),
    A("Rebecca", 1940, "Rebecca", "Daphne du Maurier"),
    A("Psycho", 1960, "Psycho", "Robert Bloch"),
    A("The Exorcist", 1973, "The Exorcist", "William Peter Blatty"),
    A("Carrie", 1976, "Carrie", "Stephen King"),
    A("Misery", 1990, "Misery", "Stephen King"),
    A("The Green Mile", 1999, "The Green Mile", "Stephen King"),
    A("Frankenstein", 1931, "Frankenstein", "Mary Shelley"),
    A("The Princess Bride", 1987, "The Princess Bride", "William Goldman"),
    A("Charlie and the Chocolate Factory", 2005, "Charlie and the Chocolate Factory", "Roald Dahl"),
    A("Matilda", 1996, "Matilda", "Roald Dahl"),
    A("Coraline", 2009, "Coraline", "Neil Gaiman"),
    A("Stardust", 2007, "Stardust", "Neil Gaiman"),
    A("The Bourne Identity", 2002, "The Bourne Identity", "Robert Ludlum"),
    A("Casino Royale", 2006, "Casino Royale", "Ian Fleming"),
    A("The Da Vinci Code", 2006, "The Da Vinci Code", "Dan Brown"),
    A("Angels & Demons", 2009, "Angels & Demons", "Dan Brown"),
    A("Bridget Jones's Diary", 2001, "Bridget Jones's Diary", "Helen Fielding"),
    A("The Devil Wears Prada", 2006, "The Devil Wears Prada", "Lauren Weisberger"),
    A("Crazy Rich Asians", 2018, "Crazy Rich Asians", "Kevin Kwan"),
    A("Where the Crawdads Sing", 2022, "Where the Crawdads Sing", "Delia Owens"),
    A("Room", 2015, "Room", "Emma Donoghue"),
    A("Gone with the Wind", 1939, "Gone with the Wind", "Margaret Mitchell"),
    A("The Notebook", 2004, "The Notebook", "Nicholas Sparks"),
    A("Blade Runner", 1982, "Do Androids Dream of Electric Sheep?", "Philip K. Dick"),
    A("A Clockwork Orange", 1971, "A Clockwork Orange", "Anthony Burgess"),
    A("Lolita", 1962, "Lolita", "Vladimir Nabokov"),
    A("Catch-22", 1970, "Catch-22", "Joseph Heller"),
    A("The Remains of the Day", 1993, "The Remains of the Day", "Kazuo Ishiguro"),
    A("Winter's Bone", 2010, "Winter's Bone", "Daniel Woodrell"),
    A("The Hunt for Red October", 1990, "The Hunt for Red October", "Tom Clancy"),
    A("The Talented Mr. Ripley", 1999, "The Talented Mr. Ripley", "Patricia Highsmith"),
    A("L.A. Confidential", 1997, "L.A. Confidential", "James Ellroy"),
    A("Trainspotting", 1996, "Trainspotting", "Irvine Welsh"),
    A("The Hours", 2002, "The Hours", "Michael Cunningham"),
    A("Sense and Sensibility", 1995, "Sense and Sensibility", "Jane Austen"),
    A("Jane Eyre", 2011, "Jane Eyre", "Charlotte Brontë"),
]


def V(  # noqa: N802
    seed: tuple[str, str, str | int], target: tuple[str, str, str | int], why: str
) -> VibeSeed:
    return VibeSeed(seed, target, why)


# Cross-media pairs that should feel alike but are NOT adaptations of each other. These are a
# first proposal: the team must agree before they count as ground truth (status stays "proposed").
VIBE_SEEDS = [
    V(
        ("movie", "Hereditary", 2018),
        ("book", "The Haunting of Hill House", "Shirley Jackson"),
        "slow-burn family dread and inherited trauma",
    ),
    V(
        ("movie", "Get Out", 2017),
        ("book", "Mexican Gothic", "Silvia Moreno-Garcia"),
        "sinister hospitality; an outsider trapped in a wealthy family's house",
    ),
    V(
        ("movie", "Parasite", 2019),
        ("book", "The White Tiger", "Aravind Adiga"),
        "class warfare and a servant's cunning climb, told as dark comedy",
    ),
    V(
        ("movie", "Spirited Away", 2001),
        ("book", "Coraline", "Neil Gaiman"),
        "a child alone in a strange spirit world with its own rules",
    ),
    V(
        ("movie", "Mad Max: Fury Road", 2015),
        ("book", "Parable of the Sower", "Octavia E. Butler"),
        "post-collapse survival journey with a found band of allies",
    ),
    V(
        ("movie", "Before Sunrise", 1995),
        ("book", "Normal People", "Sally Rooney"),
        "intimate, talky, tender connection between two young people",
    ),
    V(
        ("movie", "The Grand Budapest Hotel", 2014),
        ("book", "A Gentleman in Moscow", "Amor Towles"),
        "whimsical grand hotel and bittersweet elegance",
    ),
    V(
        ("movie", "Whiplash", 2014),
        ("book", "The Secret History", "Donna Tartt"),
        "obsessive pursuit of excellence under a charismatic, damaging mentor",
    ),
    V(
        ("movie", "Alien", 1979),
        ("book", "Annihilation", "Jeff VanderMeer"),
        "dread of an unknowable, hostile environment; a doomed expedition",
    ),
    V(
        ("movie", "Moonlight", 2016),
        ("book", "Giovanni's Room", "James Baldwin"),
        "quiet, aching queer longing and self-recognition",
    ),
    V(
        ("movie", "Knives Out", 2019),
        ("book", "The Thursday Murder Club", "Richard Osman"),
        "witty modern whodunit with an ensemble of suspects",
    ),
    V(
        ("movie", "La La Land", 2016),
        ("book", "The Night Circus", "Erin Morgenstern"),
        "dreamy, bittersweet romance between two gifted people",
    ),
    V(
        ("movie", "Coco", 2017),
        ("book", "The Midnight Library", "Matt Haig"),
        "warm, tearful second chances and what we owe those we love",
    ),
    V(
        ("movie", "Lady Bird", 2017),
        ("book", "Little Women", "Louisa May Alcott"),
        "mother-daughter friction and a restless girl growing up",
    ),
    V(
        ("movie", "Her", 2013),
        ("book", "Klara and the Sun", "Kazuo Ishiguro"),
        "tender artificial companion and quiet loneliness",
    ),
    V(
        ("movie", "Everything Everywhere All at Once", 2022),
        ("book", "The Hitchhiker's Guide to the Galaxy", "Douglas Adams"),
        "absurdist, funny, big-idea multiverse chaos",
    ),
    V(
        ("movie", "Toy Story", 1995),
        ("book", "Charlotte's Web", "E. B. White"),
        "loyal friendship and childhood warmth",
    ),
    V(
        ("movie", "Amélie", 2001),
        ("book", "Eleanor Oliphant Is Completely Fine", "Gail Honeyman"),
        "quirky, lonely heroine finding connection through small kindnesses",
    ),
    V(
        ("movie", "The Babadook", 2014),
        ("book", "We Have Always Lived in the Castle", "Shirley Jackson"),
        "eerie domestic dread and unprocessed grief",
    ),
    V(
        ("movie", "Spirited Away", 2001),
        ("book", "Howl's Moving Castle", "Diana Wynne Jones"),
        "whimsical magical-world coming-of-age with odd, memorable inhabitants",
    ),
    V(
        ("movie", "Blade Runner", 1982),
        ("book", "Neuromancer", "William Gibson"),
        "neon-soaked dystopian noir about what makes a person",
    ),
    V(
        ("movie", "Jaws", 1975),
        ("book", "Moby-Dick", "Herman Melville"),
        "obsessive hunt for a monstrous sea creature",
    ),
    V(
        ("movie", "Fight Club", 1999),
        ("book", "American Psycho", "Bret Easton Ellis"),
        "nihilistic satire of toxic masculinity and consumer culture",
    ),
    V(
        ("movie", "Dune", 2021),
        ("book", "Hyperion", "Dan Simmons"),
        "sprawling, serious space epic of politics, religion and fate",
    ),
    V(
        ("movie", "The Lord of the Rings: The Fellowship of the Ring", 2001),
        ("book", "A Wizard of Earthsea", "Ursula K. Le Guin"),
        "epic fantasy quest of a young hero facing a shadow",
    ),
    V(
        ("book", "Piranesi", "Susanna Clarke"),
        ("movie", "Pan's Labyrinth", 2006),
        "a dreamlike, labyrinthine other world and the mind that inhabits it",
    ),
    V(
        ("book", "Station Eleven", "Emily St. John Mandel"),
        ("movie", "Children of Men", 2006),
        "quiet, human-scale story after civilisation's collapse",
    ),
    V(
        ("book", "Circe", "Madeline Miller"),
        ("movie", "The Northman", 2022),
        "mythic, fate-bound saga told with a grim, lyrical intensity",
    ),
    V(
        ("book", "Klara and the Sun", "Kazuo Ishiguro"),
        ("movie", "Ex Machina", 2015),
        "an artificial being observed by humans; quiet unease about what it feels",
    ),
    V(
        ("book", "The Night Circus", "Erin Morgenstern"),
        ("movie", "The Prestige", 2006),
        "rival magicians, secrecy and enchanting spectacle with a cost",
    ),
    V(
        ("book", "Normal People", "Sally Rooney"),
        ("movie", "Past Lives", 2023),
        "restrained, aching love across time and missed chances",
    ),
    V(
        ("book", "Gilead", "Marilynne Robinson"),
        ("movie", "Paterson", 2016),
        "contemplative attention to ordinary life, faith and small beauties",
    ),
]


@dataclass(frozen=True)
class GoldPair:
    book_ol_id: str
    movie_tmdb_id: int


def read_gold_adaptations(path: Path = GOLD_ADAPTATIONS_PATH) -> list[GoldPair]:
    """Return the gold adaptation pairs, or [] if the file doesn't exist yet."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [GoldPair(r["book_ol_id"], int(r["movie_tmdb_id"])) for r in csv.DictReader(f)]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _exact(work_note: str | None) -> bool:
    return work_note is None


def draft_adaptations(
    tmdb_client: CachedClient,
    ol_client: CachedClient,
    wikidata_pairs: set[tuple[int, str]],
    seeds: list[AdaptationSeed] = ADAPTATION_SEEDS,
) -> tuple[list[dict[str, object]], list[str]]:
    """Resolve the seeds to ids. Returns (csv rows, rejected seeds with the reason)."""
    rows: list[dict[str, object]] = []
    rejected: list[str] = []
    seen: set[tuple[str, int]] = set()
    for seed in seeds:
        label = f"{seed.movie_title} ({seed.movie_year}) / {seed.book_title}"
        try:
            movie = tmdb.find_movie(tmdb_client, seed.movie_title, seed.movie_year)
            book = openlibrary.find_book(ol_client, seed.book_title, seed.book_author)
        except SkipWork as exc:
            rejected.append(f"{label}: {exc}")
            continue
        if not (_exact(movie.match_note) and _exact(book.match_note)):
            rejected.append(f"{label}: not an exact match ({movie.match_note or book.match_note})")
            continue
        key = (book.external_id, int(movie.external_id))
        if key in seen:
            rejected.append(f"{label}: duplicate of an earlier pair")
            continue
        seen.add(key)
        rows.append(
            {
                "book_ol_id": book.external_id,
                "movie_tmdb_id": movie.external_id,
                "book_title": book.title,
                "book_author": "; ".join(book.creators[:1]),
                "movie_title": movie.title,
                "movie_year": movie.year,
                "in_wikidata": (int(movie.external_id), book.external_id) in wikidata_pairs,
            }
        )
    return rows, rejected


def _resolve(
    ref: tuple[str, str, str | int], tmdb_client: CachedClient, ol_client: CachedClient
) -> tuple[str, str]:
    """Return (ref like 'tmdb:694' or 'ol:OL1W', display title); raises SkipWork if inexact."""
    kind, title, extra = ref
    if kind == "movie":
        movie = tmdb.find_movie(tmdb_client, title, int(extra))
        if movie.match_note:
            raise SkipWork(movie.match_note)
        return f"tmdb:{movie.external_id}", movie.title
    book = openlibrary.find_book(ol_client, title, str(extra))
    if book.match_note:
        raise SkipWork(book.match_note)
    return f"ol:{book.external_id}", book.title


def draft_vibes(
    tmdb_client: CachedClient, ol_client: CachedClient, seeds: list[VibeSeed] = VIBE_SEEDS
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    rejected: list[str] = []
    for seed in seeds:
        label = f"{seed.seed[1]} -> {seed.target[1]}"
        try:
            seed_ref, seed_title = _resolve(seed.seed, tmdb_client, ol_client)
            target_ref, target_title = _resolve(seed.target, tmdb_client, ol_client)
        except SkipWork as exc:
            rejected.append(f"{label}: {exc}")
            continue
        rows.append(
            {
                "seed_ref": seed_ref,
                "target_ref": target_ref,
                "seed_title": seed_title,
                "target_title": target_title,
                "why": seed.why,
                "status": "proposed",
            }
        )
    return rows, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft the gold evaluation files.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft-adaptations", action="store_true")
    mode.add_argument("--draft-vibes", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with tmdb.make_client() as t, openlibrary.make_client() as o:
        if args.draft_adaptations:
            with wikidata.make_client() as w:
                pairs = set(wikidata.fetch_adaptation_pairs(w))
            rows, rejected = draft_adaptations(t, o, pairs)
            write_csv(GOLD_ADAPTATIONS_PATH, ADAPTATION_FIELDS, rows)
            covered = sum(bool(r["in_wikidata"]) for r in rows)
            logger.info(
                "Wrote %d pairs to %s (%d in Wikidata, %d not)",
                len(rows),
                GOLD_ADAPTATIONS_PATH,
                covered,
                len(rows) - covered,
            )
        else:
            rows, rejected = draft_vibes(t, o)
            write_csv(GOLD_VIBES_PATH, VIBE_FIELDS, rows)
            logger.info("Wrote %d proposed pairs to %s", len(rows), GOLD_VIBES_PATH)
    for line in rejected:
        logger.warning("NOT INCLUDED %s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
