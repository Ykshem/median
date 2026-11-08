"""Anthropic + OpenAI wrappers with retries, usage logging and the daily spend guard.

Every API call in the project goes through this module so that `llm_usage` is complete and
`DAILY_USD_LIMIT` is enforced. Clients are created lazily, so importing never needs API keys.
Bulk jobs (vibe cards, link checks) use the Message Batches API helpers at the bottom (half price).
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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


# Claude Haiku 5.5 thinks by default, and thinking tokens count against max_tokens and are billed as
# output. Extraction-style jobs (vibe cards, link checks) gain nothing from it: measured on real
# works, 3 of 3 cards were valid at ~300 output tokens with thinking off, but with it on a card
# used 580-670 tokens and one was cut off. Pass this as `thinking=` for such calls.
THINKING_OFF: dict[str, str] = {"type": "disabled"}


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
    thinking: dict[str, Any] | None = None,
    output_config: dict[str, Any] | None = None,
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
    if thinking is not None:
        kwargs["thinking"] = thinking
    if output_config is not None:
        kwargs["output_config"] = output_config
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
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None


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
        if request.thinking is not None:
            params["thinking"] = request.thinking
        if request.output_config is not None:
            params["output_config"] = request.output_config
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
    on_result: Callable[[BatchResult], None] | None = None,
) -> list[BatchResult]:
    """Submit (or resume), wait for, and collect one batch. Returns results in arrival order.

    `on_result`, if given, is called for each result once the batch has been collected.
    """
    batch_id = submit_batch(
        conn,
        purpose=purpose,
        tag=tag,
        model=model,
        requests=requests,
        estimated_usd=estimated_usd,
    )
    wait_for_batch(batch_id, poll_seconds=poll_seconds, sleep=sleep)
    results = collect_batch(conn, batch_id, purpose=purpose, model=model)
    if on_result is not None:
        for result in results:
            on_result(result)
    return results


# Regular API, many requests at once (full price, but no queue) ------------------------------------

DEFAULT_WORKERS = 6
PARALLEL_ATTEMPTS = 8  # more patient than complete(): many requests share one rate limit


def _is_fatal(exc: Exception) -> bool:
    """True for 4xx errors that retrying cannot fix (bad request, no credit, bad key)."""
    return (
        isinstance(exc, anthropic.APIStatusError)
        and 400 <= exc.status_code < 500
        and exc.status_code not in (408, 409, 429)
    )


def _call_one(client: Any, model: str, request: BatchRequest) -> BatchResult:
    """Make one Messages call (in a worker thread) and return it as a BatchResult.

    Transient failures are retried with backoff. A request that still fails comes back as an
    `errored` result so the caller can retry it; a fatal 4xx is raised to stop the whole run.
    Touches no database: workers only talk to the API.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    }
    for name in ("system", "thinking", "output_config"):
        value = getattr(request, name)
        if value is not None:
            kwargs[name] = value
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(_TRANSIENT),
            stop=stop_after_attempt(PARALLEL_ATTEMPTS),
            wait=_WAIT,
            reraise=True,
        ):
            with attempt:
                response = client.messages.create(**kwargs)
    except Exception as exc:
        if _is_fatal(exc):
            raise
        return BatchResult(request.custom_id, "errored", error=f"{type(exc).__name__}: {exc}")
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return BatchResult(
        request.custom_id,
        "succeeded",
        text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


def run_parallel(
    conn: sqlite3.Connection,
    *,
    purpose: str,
    model: str,
    requests: list[BatchRequest],
    estimated_usd: float = 0.0,
    workers: int = DEFAULT_WORKERS,
    on_result: Callable[[BatchResult], None] | None = None,
    tag: str | None = None,
    poll_seconds: float | None = None,
    sleep: Callable[[float], None] | None = None,
) -> list[BatchResult]:
    """Run the requests concurrently on the regular Messages API; return results in arrival order.

    A drop-in for `run_batch` (`tag`, `poll_seconds` and `sleep` are accepted and ignored) that
    costs twice as much but has no queue. Only this thread touches the database: each result is
    logged at the regular price and passed to `on_result` as it arrives, so an interrupted run
    keeps its work and its spend record. The daily budget is checked before every request is sent;
    when it trips, requests already in flight are collected and logged first, then BudgetExceeded
    is raised.
    """
    if not requests:
        return []
    check_budget(conn, estimated_usd)
    client = _anthropic_client()
    results: list[BatchResult] = []
    pending = iter(requests)
    in_flight: set[Future[BatchResult]] = set()

    def finish(result: BatchResult) -> None:
        if result.status == "succeeded":
            log_usage(
                conn,
                purpose=purpose,
                model=model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        results.append(result)
        if on_result is not None:
            on_result(result)

    with ThreadPoolExecutor(max_workers=workers) as pool:

        def launch() -> bool:
            request = next(pending, None)
            if request is None:
                return False
            check_budget(conn)
            in_flight.add(pool.submit(_call_one, client, model, request))
            return True

        try:
            for _ in range(workers):
                if not launch():
                    break
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.discard(future)
                    finish(future.result())
                    launch()
        finally:
            # Whatever stopped us, requests already sent are billed: collect and log them.
            for future in list(in_flight):
                with contextlib.suppress(Exception):
                    finish(future.result())
    return results
