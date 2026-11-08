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
