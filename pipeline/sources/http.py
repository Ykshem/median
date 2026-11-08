"""A small JSON-over-HTTP client with an on-disk response cache and a request throttle.

Raw responses are cached under `data/cache/<source>/` so reruns are free and fast. Cache files hold
only the request path/params and the response, never headers, so credentials can't leak into them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from median.config import CACHE_DIR

logger = logging.getLogger(__name__)

MAX_RATE_LIMIT_RETRIES = 3


class CachedClient:
    """GET JSON from one API: cache hit costs nothing; a miss is throttled and 429-aware."""

    def __init__(
        self,
        source: str,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        min_interval: float = 0.3,
        timeout: float = 20.0,
        cache_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self._cache_dir = (cache_dir or CACHE_DIR) / source
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.network_calls = 0
        self._http = httpx.Client(
            base_url=base_url,
            headers=headers or {},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> CachedClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        key = json.dumps([path, sorted((params or {}).items())], default=str)
        return self._cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Return the parsed JSON for GET `path`, from the cache when possible."""
        cache_file = self._cache_path(path, params)
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["response"]

        self._throttle()
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            response = self._http.get(path, params=params)
            self._last_request = self._clock()
            self.network_calls += 1
            if response.status_code != 429:
                break
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise RuntimeError(f"{self.source}: still rate-limited after retries for {path}")
            try:
                delay = float(response.headers.get("Retry-After", ""))
            except ValueError:
                delay = float(2 ** (attempt + 1))
            logger.warning("%s: HTTP 429, waiting %.0fs", self.source, delay)
            self._sleep(delay)
        response.raise_for_status()
        data = response.json()

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"path": path, "params": params, "response": data}), "utf-8")
        tmp.replace(cache_file)
        return data
