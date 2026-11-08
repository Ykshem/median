"""Shared prompt helpers: load a template and fill it safely from untrusted catalog text."""

from __future__ import annotations

import re
from collections.abc import Mapping

from median.config import PROMPTS_DIR

PLACEHOLDER = re.compile(r"\{(\w+)\}")
DATA_TAG = re.compile(r"</?\s*work_data\s*>", re.IGNORECASE)


def load_prompt(name: str) -> str:
    """Return the text of median/prompts/<name>.md."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def untrusted(text: str) -> str:
    """Remove any work_data tags from catalog text so it can't close the data block early."""
    return DATA_TAG.sub("", text)


def fill(template: str, values: Mapping[str, str]) -> str:
    """Fill `{placeholders}` in a single pass, so text inside a value is never expanded again.

    Values are passed through `untrusted` first. Unknown placeholders are left as written.
    """
    safe = {key: untrusted(value) for key, value in values.items()}
    return PLACEHOLDER.sub(lambda m: safe.get(m.group(1), m.group(0)), template)
