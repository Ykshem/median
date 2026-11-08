"""Core data types: the pydantic VibeCard and plain dataclasses mirroring DB rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

MediaType = Literal["movie", "book", "tv", "music"]
LinkType = Literal["adaptation_of", "same_franchise", "same_creator"]
ShelfName = Literal["want", "in_progress", "finished", "favorite"]


class VibeCard(BaseModel):
    """Media-neutral description of how a story feels (ARCHITECTURE.md section 4)."""

    logline: str  # 1 spoiler-free sentence about the story, not the medium
    themes: list[str] = Field(min_length=3, max_length=6)
    mood: list[str] = Field(min_length=3, max_length=5)
    tone: str
    setting: str
    pacing: Literal["slow", "medium", "fast"]
    intensity: Literal["low", "medium", "high"]
    audience: Literal["children", "young_adult", "adult"]
    hooks: list[str] = Field(min_length=2, max_length=4)

    def to_text(self) -> str:
        """Return the flattened text that gets embedded (`vibe_cards.card_text`)."""
        return (
            f"{self.logline} "
            f"Themes: {', '.join(self.themes)}. Mood: {', '.join(self.mood)}. "
            f"Tone: {self.tone}. Setting: {self.setting}. Pacing: {self.pacing}. "
            f"Intensity: {self.intensity}. Audience: {self.audience}. "
            f"Hooks: {', '.join(self.hooks)}."
        )


@dataclass(frozen=True)
class Work:
    id: int
    media_type: MediaType
    title: str
    year: int | None = None
    creators: list[str] = field(default_factory=list)
    synopsis: str | None = None
    subjects: list[str] = field(default_factory=list)
    popularity: float | None = None
    cover_url: str | None = None


@dataclass(frozen=True)
class Link:
    from_work: int
    to_work: int
    link_type: LinkType
    source: str
    confidence: float = 1.0
    verified: bool = False


@dataclass(frozen=True)
class Persona:
    id: int
    name: str
    description: str | None = None
    taste_vector: bytes | None = None  # float32[1536] blob; None until >= 3 signals


@dataclass(frozen=True)
class ShelfItem:
    persona_id: int
    work_id: int
    shelf: ShelfName
    rating: int | None = None  # 1..10
    note: str | None = None


@dataclass(frozen=True)
class Recommendation:
    """A ranked candidate. `score` is the weighted sum of the four components, each in 0..1."""

    work_id: int
    score: float
    reasons: list[str] = field(default_factory=list)  # machine-made, shown in the UI and explainer
    link: float = 0.0  # strength of the strongest verified link to the seed
    vibe: float = 0.0  # 1 - cosine distance to the seed (or query) vector
    taste: float = 0.0  # 1 - cosine distance to the persona taste vector (0 without one)
    popularity: float = 0.0  # percentile within the media type
