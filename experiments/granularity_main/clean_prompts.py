"""Derive intent / workflow / action instructions from ``prompt_specs``.

Intent: AndroidWorld ``task.goal`` only (no appended suffix).

Workflow / action: append the filled template from the canonical spec for that
task. Placeholders ``{name}`` are replaced only when the key exists in
``task.params`` and maps to a scalar value (string, int, float, bool).
Structured params (e.g. SQLite rows) never appear in templates — those tasks use
generic step text and rely on ``task.goal`` for all literal values.
"""

from __future__ import annotations

from typing import Any

from experiments.granularity_main.prompt_specs import CANONICAL_SPECS

GRANULARITY_LEVELS = ("intent", "workflow", "action")


def _fill_scalars(template: str, params: dict[str, Any]) -> str:
    """Replace {key} for keys whose values are scalars."""

    out = template
    for key, val in params.items():
        if not isinstance(key, str):
            continue
        if isinstance(val, (dict, list)):
            continue
        placeholder = "{" + key + "}"
        if placeholder in out:
            out = out.replace(placeholder, str(val))
    return out


def granularity_suffix(task_class: str, granularity: str, params: dict[str, Any]) -> str:
    """Return text appended after ``task.goal`` for workflow/action; empty for intent."""

    if granularity not in GRANULARITY_LEVELS:
        raise ValueError(
            f"Unsupported granularity={granularity!r}; expected one of {GRANULARITY_LEVELS}"
        )
    if granularity == "intent":
        return ""
    spec = CANONICAL_SPECS.get(task_class)
    if spec is None:
        raise KeyError(f"No canonical spec for task_class={task_class!r}")
    raw = (
        spec.workflow_template
        if granularity == "workflow"
        else spec.action_template
    )
    return _fill_scalars(raw, params).strip()


def build_agent_goal(task_goal: str, task_class: str, granularity: str, params: dict[str, Any]) -> str:
    """Full instruction string passed to the agent (T3A goal)."""

    suffix = granularity_suffix(task_class, granularity, params)
    if not suffix:
        return task_goal
    return task_goal.strip() + "\n\n" + suffix
