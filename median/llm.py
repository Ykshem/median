"""Anthropic + OpenAI wrappers with retries, usage logging and the daily spend guard.

Every API call in the project goes through this module so that `llm_usage` is complete and
`DAILY_USD_LIMIT` is enforced. Clients are created lazily, so importing never needs API keys.
Bulk jobs (vibe cards, link checks) will add Message Batches helpers here in M1.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from typing import Any

import anthropic
import openai
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from median.config import (
    BATCH_DISCOUNT,
    EMBED_DIMS,
    EMBED_MODEL,
    PRICES_PER_MTOK,
    get_settings,
)

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 256
MAX_ATTEMPTS = 3
_WAIT = wait_exponential(multiplier=1, max=20)  # tests replace this with wait_none()
_TRANSIENT = (
    anthropic.APIConnectionError,  # includes timeouts
    anthropic.RateLimitError,
    anthropic.InternalServerError,  # 5xx, including 529 overloaded
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class BudgetExceeded(RuntimeError):
    """Raised instead of making an API call once today's spend is over DAILY_USD_LIMIT."""


# Cost and budget ---------------------------------------------------------------------------------


def cost_usd(model: str, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
    """Return the USD cost of one call. Raises ValueError if the model has no configured price."""
    try:
        price_in, price_out = PRICES_PER_MTOK[model]
    except KeyError:
        raise ValueError(f"No price configured for model {model!r}; add it to config.py") from None
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


def log_usage(
    conn: sqlite3.Connection,
    *,
    purpose: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    batch: bool = False,
) -> float:
    """Insert a row into `llm_usage` and return its cost in USD."""
    cost = cost_usd(model, input_tokens, output_tokens, batch=batch)
    conn.execute(
        "INSERT INTO llm_usage (purpose, model, input_tokens, output_tokens, cost_usd) "
        "VALUES (?, ?, ?, ?, ?)",
        (purpose, model, input_tokens, output_tokens, cost),
    )
    conn.commit()
    logger.info("LLM usage purpose=%s model=%s cost=$%.5f", purpose, model, cost)
    return cost


def spent_today(conn: sqlite3.Connection) -> float:
    """Return USD spent today (UTC date, matching the CURRENT_TIMESTAMP column default)."""
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM llm_usage "
        "WHERE date(created_at) = date('now')"
    ).fetchone()
    return float(row["total"])


def check_budget(conn: sqlite3.Connection, estimated: float = 0.0) -> None:
    """Raise BudgetExceeded if today's spend plus `estimated` would exceed DAILY_USD_LIMIT."""
    limit = get_settings().daily_usd_limit
    spent = spent_today(conn)
    if spent + estimated > limit:
        raise BudgetExceeded(
            f"Daily LLM limit reached: ${spent:.2f} spent today + ${estimated:.2f} estimated "
            f"exceeds the ${limit:.2f} limit. Raise DAILY_USD_LIMIT in .env to continue."
        )


# Clients -----------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _anthropic_client() -> anthropic.Anthropic:
    key = get_settings().anthropic_api_key
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return anthropic.Anthropic(api_key=key, max_retries=0)  # retries are handled by tenacity here


@lru_cache(maxsize=1)
def _openai_client() -> openai.OpenAI:
    key = get_settings().openai_api_key
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return openai.OpenAI(api_key=key, max_retries=0)


def _retrying() -> Retrying:
    return Retrying(
        retry=retry_if_exception_type(_TRANSIENT),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=_WAIT,
        reraise=True,
    )


# Calls -------------------------------------------------------------------------------------------


def complete(
    conn: sqlite3.Connection,
    *,
    purpose: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 1024,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    """Call the Anthropic Messages API and return the response (non-streaming).

    Checks the daily budget first, retries transient failures up to 3 times, then logs token usage.
    Pass a model ID from `median.config`, never a literal.
    """
    check_budget(conn)
    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools
    client = _anthropic_client()
    for attempt in _retrying():
        with attempt:
            response = client.messages.create(**kwargs)
    log_usage(
        conn,
        purpose=purpose,
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return response


def embed(conn: sqlite3.Connection, texts: list[str], *, purpose: str) -> list[list[float]]:
    """Return one `EMBED_DIMS`-length vector per input text, in order.

    Sends at most 256 texts per request and logs one usage row per request.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        check_budget(conn)
        client = _openai_client()
        for attempt in _retrying():
            with attempt:
                response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        log_usage(
            conn,
            purpose=purpose,
            model=EMBED_MODEL,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=0,
        )
        for item in sorted(response.data, key=lambda d: d.index):
            if len(item.embedding) != EMBED_DIMS:
                raise ValueError(f"Expected {EMBED_DIMS}-dim embeddings, got {len(item.embedding)}")
            vectors.append(list(item.embedding))
    return vectors
