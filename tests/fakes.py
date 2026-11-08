"""Scripted fake API clients. No network, no real keys."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _Script:
    """Returns/raises the scripted items in order and records every call's kwargs."""

    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.calls: list[dict[str, Any]] = []

    def next(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def anthropic_response(text: str = "ok", input_tokens: int = 100, output_tokens: int = 20) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeAnthropic:
    def __init__(self, items: list[Any]) -> None:
        self._script = _Script(items)
        self.messages = SimpleNamespace(create=lambda **kw: self._script.next(kw))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._script.calls


def embedding_response(n: int, dims: int = 1536, prompt_tokens: int = 50) -> Any:
    data = [SimpleNamespace(index=i, embedding=[float(i)] * dims) for i in range(n)]
    return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=prompt_tokens))


class FakeOpenAI:
    def __init__(self, items: list[Any]) -> None:
        self._script = _Script(items)
        self.embeddings = SimpleNamespace(create=lambda **kw: self._script.next(kw))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._script.calls


class FakeBatchClient:
    """A fake Anthropic client for the Batches API.

    `answer(request)` gets each submitted request dict ({"custom_id", "params"}) and returns the
    reply text, a `(text, input_tokens, output_tokens)` tuple, or None for an errored request.
    `statuses` is the sequence of `processing_status` values that retrieve() reports.
    """

    def __init__(self, answer, statuses=("in_progress", "ended")) -> None:
        self._answer = answer
        self._statuses = list(statuses)
        self.created: list[list[dict[str, Any]]] = []
        self.retrieves = 0
        self._results: dict[str, list[Any]] = {}
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(
                create=self._create, retrieve=self._retrieve, results=self._get_results
            )
        )

    def _create(self, *, requests: list[dict[str, Any]]) -> Any:
        batch_id = f"msgbatch_{len(self.created) + 1}"
        self.created.append(requests)
        self._results[batch_id] = [self._item(r) for r in requests]
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def _item(self, request: dict[str, Any]) -> Any:
        reply = self._answer(request)
        if reply is None:
            error = SimpleNamespace(model_dump=lambda: {"type": "api_error", "message": "boom"})
            result = SimpleNamespace(type="errored", error=error)
        else:
            text, tokens_in, tokens_out = reply if isinstance(reply, tuple) else (reply, 400, 200)
            message = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
            )
            result = SimpleNamespace(type="succeeded", message=message)
        return SimpleNamespace(custom_id=request["custom_id"], result=result)

    def _retrieve(self, batch_id: str) -> Any:
        self.retrieves += 1
        status = self._statuses[min(self.retrieves - 1, len(self._statuses) - 1)]
        counts = SimpleNamespace(succeeded=0, errored=0, processing=1)
        return SimpleNamespace(id=batch_id, processing_status=status, request_counts=counts)

    def _get_results(self, batch_id: str) -> list[Any]:
        return self._results[batch_id]


class FakeConcurrentClient:
    """A thread-safe fake Anthropic client for regular (non-batch) calls made in parallel.

    `answer(kwargs)` gets the keyword arguments of `messages.create` and returns the reply text, a
    `(text, input_tokens, output_tokens)` tuple, or an Exception instance to raise.
    """

    def __init__(self, answer, delay: float = 0.0) -> None:
        import threading

        self._answer = answer
        self._delay = delay
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.threads: set[int] = set()
        self._get_ident = threading.get_ident
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        import time

        with self._lock:
            self.calls.append(kwargs)
            self.threads.add(self._get_ident())
        if self._delay:
            time.sleep(self._delay)
        reply = self._answer(kwargs)
        if isinstance(reply, Exception):
            raise reply
        text, tokens_in, tokens_out = reply if isinstance(reply, tuple) else (reply, 400, 200)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
        )
