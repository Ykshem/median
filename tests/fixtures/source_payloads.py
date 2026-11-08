"""Small hand-written API payloads shaped like TMDB and Open Library responses."""

from __future__ import annotations

from typing import Any


def tmdb_details(
    movie_id: int = 694,
    title: str = "The Shining",
    date: str = "1980-05-23",
    overview: str = "A writer takes a winter caretaker job at an isolated hotel with his family.",
) -> dict[str, Any]:
    return {
        "id": movie_id,
        "title": title,
        "release_date": date,
        "overview": overview,
        "poster_path": "/poster.jpg",
        "genres": [{"id": 27, "name": "Horror"}, {"id": 53, "name": "Thriller"}],
        "keywords": {"keywords": [{"id": 1, "name": "hotel"}, {"id": 2, "name": "isolation"}]},
        "credits": {
            "crew": [
                {"name": "Stanley Kubrick", "job": "Director"},
                {"name": "Jan Harlan", "job": "Executive Producer"},
            ]
        },
        "external_ids": {"imdb_id": "tt0081505"},
    }


def tmdb_search(*results: tuple[int, str, str]) -> dict[str, Any]:
    return {
        "results": [
            {"id": i, "title": t, "release_date": d, "popularity": 10.0 - n}
            for n, (i, t, d) in enumerate(results)
        ]
    }


def ol_search(*docs: dict[str, Any]) -> dict[str, Any]:
    return {"docs": list(docs)}


def ol_doc(
    key: str = "/works/OL81634W",
    title: str = "The Shining",
    authors: tuple[str, ...] = ("Stephen King",),
    year: int = 1977,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "author_name": list(authors),
        "first_publish_year": year,
        "cover_i": 12345,
    }


def ol_work(
    description: Any = "A father takes his family to an empty mountain hotel.",
) -> dict[str, Any]:
    work: dict[str, Any] = {"title": "The Shining", "subjects": ["Horror", "Hotels", "Fiction"]}
    if description is not None:
        work["description"] = description
    return work
