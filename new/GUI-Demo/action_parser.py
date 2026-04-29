"""Parse Doubao GUI action text into a structured representation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.I | re.S)
THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?:\n\s*Action\s*:|$)", re.I | re.S)
CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.S)
POINT_RE = re.compile(r"<(?:start_)?point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</(?:start_)?point>")


@dataclass(frozen=True)
class ParsedAction:
    """One parsed Doubao GUI action."""

    thought: str
    action_text: str
    action_name: str
    point: tuple[float, float] | None = None
    start_point: tuple[float, float] | None = None
    end_point: tuple[float, float] | None = None
    content: str | None = None
    direction: str | None = None
    app_name: str | None = None


def parse_doubao_response(response: str) -> ParsedAction:
    """Parse a `Thought: ... Action: ...` response."""
    thought = _extract_thought(response)
    action_text = _extract_action_text(response)
    match = CALL_RE.match(action_text)
    if not match:
        raise ValueError(f"Could not parse action call: {action_text}")

    action_name = match.group(1).strip()
    args = match.group(2)
    kwargs = _parse_call_kwargs(action_text)
    return ParsedAction(
        thought=thought,
        action_text=action_text,
        action_name=action_name,
        point=_extract_point_arg(args, "point", kwargs),
        start_point=_extract_point_arg(args, "start_point", kwargs),
        end_point=_extract_point_arg(args, "end_point", kwargs),
        content=_extract_string_arg(args, "content", kwargs),
        direction=_extract_string_arg(args, "direction", kwargs),
        app_name=_extract_string_arg(args, "app_name", kwargs),
    )


def _extract_thought(response: str) -> str:
    match = THOUGHT_RE.search(response)
    return match.group(1).strip() if match else ""


def _extract_action_text(response: str) -> str:
    match = ACTION_RE.search(response)
    if not match:
        direct_call = _direct_action_call(response)
        if direct_call:
            return direct_call
        fallback = _fallback_action_text(response)
        if fallback:
            return fallback
        raise ValueError(f"Missing Action field in response: {response}")
    text = match.group(1).strip()
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    return text.splitlines()[0].strip()


def _direct_action_call(response: str) -> str | None:
    """Accept model outputs where the whole response is already an action call."""
    text = response.strip()
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for line in text.splitlines():
        candidate = line.strip()
        if CALL_RE.match(candidate):
            return candidate
    return None


def _fallback_action_text(response: str) -> str | None:
    """Recover common non-compliant Doubao GUI action wording."""
    lowered = response.lower()
    point = _first_raw_point(response)
    if "finished" in lowered or "task complete" in lowered or "task is complete" in lowered:
        return "finished(content='complete')"
    if point and ("long press" in lowered or "long-press" in lowered or "longpress" in lowered):
        return f"long_press(point='{point}')"
    if point and ("click" in lowered or "tap" in lowered or "press" in lowered):
        return f"click(point='{point}')"
    return None


def _first_raw_point(text: str) -> str | None:
    match = POINT_RE.search(text)
    if not match:
        return None
    return f"<point>{match.group(1)} {match.group(2)}</point>"


def _parse_call_kwargs(action_text: str) -> dict[str, object]:
    """Parse keyword arguments without regex truncating escaped quotes."""
    try:
        tree = ast.parse(action_text, mode="eval")
    except SyntaxError:
        return {}
    call = tree.body
    if not isinstance(call, ast.Call):
        return {}

    kwargs: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except (SyntaxError, ValueError):
            continue
    return kwargs


def _extract_string_arg(args: str, name: str, kwargs: dict[str, object] | None = None) -> str | None:
    if kwargs and name in kwargs:
        value = kwargs[name]
        return str(value) if value is not None else None
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(['\"])((?:\\.|(?!\1).)*)\1", args, re.S)
    if not match:
        return None
    literal = match.group(0).split("=", 1)[1].strip()
    try:
        return str(ast.literal_eval(literal))
    except (SyntaxError, ValueError):
        return match.group(2)


def _extract_point_arg(
    args: str,
    name: str,
    kwargs: dict[str, object] | None = None,
) -> tuple[float, float] | None:
    value = _extract_string_arg(args, name, kwargs)
    if not value:
        return None
    match = POINT_RE.search(value)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))
