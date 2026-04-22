"""JSON helpers for parsing model outputs."""

from __future__ import annotations

import json
import re
from typing import Any


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.I)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extracts the first JSON object from a model response."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    stripped = text.strip()
    for candidate in _candidate_json_strings(stripped):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found in model response.")


def _candidate_json_strings(text: str) -> list[str]:
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in _CODE_FENCE_RE.finditer(text))

    first_brace = text.find("{")
    if first_brace >= 0:
        decoder = json.JSONDecoder()
        try:
            _, end = decoder.raw_decode(text[first_brace:])
            candidates.append(text[first_brace : first_brace + end])
        except json.JSONDecodeError:
            candidates.append(_balanced_object_slice(text[first_brace:]))
    return [candidate for candidate in candidates if candidate]


def _balanced_object_slice(text: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return text
