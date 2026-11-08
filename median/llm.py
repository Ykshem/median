"""Anthropic + OpenAI wrappers with retries, usage logging and the daily spend guard.

Every API call in the project goes through this module so that `llm_usage` is complete and
`DAILY_USD_LIMIT` is enforced. Clients are created lazily, so importing never needs API keys.
Bulk jobs (vibe cards, link checks) use the Message Batches API helpers at the bottom (half price).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
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
from median.db import get_meta, set_meta

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


# Message Batches API (bulk jobs, 50% off) ---------------------------------------------------------

MAX_BATCH_REQUESTS = 10_000  # the API allows 100k; smaller batches finish sooner and fail smaller
BATCH_POLL_SECONDS = 30.0
BATCH_TIMEOUT_SECONDS = 24 * 3600.0  # batches expire after 24 hours


def sleep_for(seconds: float) -> None:
    """Sleep. Looked up at call time (not bound as a default), so tests can patch `time.sleep`."""
    time.sleep(seconds)


@dataclass(frozen=True)
class BatchRequest:
    """One request in a batch. `custom_id` must match [A-Za-z0-9_-]{1,64}."""

    custom_id: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    system: str | None = None


@dataclass(frozen=True)
class BatchResult:
    custom_id: str
    status: str  # succeeded | errored | canceled | expired
    text: str | None = None  # the reply text, when succeeded
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def _batch_key(purpose: str, tag: str) -> str:
    return f"batch:{purpose}:{tag}"


def submit_batch(
    conn: sqlite3.Connection,
    *,
    purpose: str,
    tag: str,
    model: str,
    requests: list[BatchRequest],
    estimated_usd: float = 0.0,
) -> str:
    """Submit a batch and return its id, or return the id already recorded for (purpose, tag).

    The id is stored in `meta`, so rerunning a step resumes the same batch instead of paying twice.
    Not retried on failure: a retry after a lost response could create (and bill) a duplicate.
    """
    key = _batch_key(purpose, tag)
    existing = get_meta(conn, key)
    if existing:
        logger.info("Resuming existing batch %s (%s/%s)", existing, purpose, tag)
        return existing
    if not requests:
        raise ValueError("Cannot submit an empty batch")
    if len(requests) > MAX_BATCH_REQUESTS:
        raise ValueError(f"Batch has {len(requests)} requests; split it (max {MAX_BATCH_REQUESTS})")
    check_budget(conn, estimated=estimated_usd)
    payload = []
    for request in requests:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
        }
        if request.system is not None:
            params["system"] = request.system
        payload.append({"custom_id": request.custom_id, "params": params})
    batch = _anthropic_client().messages.batches.create(requests=payload)
    set_meta(conn, key, batch.id)
    conn.commit()
    logger.info("Submitted batch %s: %d requests (%s/%s)", batch.id, len(requests), purpose, tag)
    return batch.id


def wait_for_batch(
    batch_id: str,
    *,
    poll_seconds: float = BATCH_POLL_SECONDS,
    timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = sleep_for,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Poll until the batch has ended and return it. Raises TimeoutError if it takes too long."""
    client = _anthropic_client()
    started = clock()
    while True:
        for attempt in _retrying():
            with attempt:
                batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        if clock() - started > timeout_seconds:
            raise TimeoutError(f"Batch {batch_id} still {batch.processing_status} after timeout")
        counts = batch.request_counts
        logger.info(
            "Batch %s %s: %d done, %d processing",
            batch_id,
            batch.processing_status,
            counts.succeeded + counts.errored,
            counts.processing,
        )
        sleep(poll_seconds)


def _error_detail(error: Any) -> str | None:
    """Return a JSON string describing a batch request error (None when there is none)."""
    if error is None:
        return None
    dump = getattr(error, "model_dump", None)
    return json.dumps(dump() if dump else str(error))


def collect_batch(
    conn: sqlite3.Connection, batch_id: str, *, purpose: str, model: str
) -> list[BatchResult]:
    """Return every result of an ended batch, logging its usage once at the batch price.

    The usage is written as one `llm_usage` row per batch, and a `meta` flag stops a rerun from
    logging it a second time.
    """
    client = _anthropic_client()
    results: list[BatchResult] = []
    for attempt in _retrying():
        with attempt:
            results = []
            for item in client.messages.batches.results(batch_id):
                outcome = item.result
                if outcome.type == "succeeded":
                    message = outcome.message
                    text = "".join(
                        b.text for b in message.content if getattr(b, "type", None) == "text"
                    )
                    results.append(
                        BatchResult(
                            item.custom_id,
                            "succeeded",
                            text,
                            message.usage.input_tokens,
                            message.usage.output_tokens,
                        )
                    )
                else:
                    detail = _error_detail(getattr(outcome, "error", None))
                    results.append(BatchResult(item.custom_id, outcome.type, error=detail))
    flag = f"batch_logged:{batch_id}"
    if get_meta(conn, flag) is None:
        log_usage(
            conn,
            purpose=purpose,
            model=model,
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
            batch=True,
        )
        set_meta(conn, flag, "1")
        conn.commit()
    return results


def run_batch(
    conn: sqlite3.Connection,
    *,
    purpose: str,
    tag: str,
    model: str,
    requests: list[BatchRequest],
    estimated_usd: float = 0.0,
    poll_seconds: float = BATCH_POLL_SECONDS,
    sleep: Callable[[float], None] = sleep_for,
) -> list[BatchResult]:
    """Submit (or resume), wait for, and collect one batch. Returns results in arrival order."""
    batch_id = submit_batch(
        conn,
        purpose=purpose,
        tag=tag,
        model=model,
        requests=requests,
        estimated_usd=estimated_usd,
    )
    wait_for_batch(batch_id, poll_seconds=poll_seconds, sleep=sleep)
    return collect_batch(conn, batch_id, purpose=purpose, model=model)
