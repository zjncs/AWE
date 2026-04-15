"""Study B: perturbations applied on top of clean prompts.

Clean prompts live in ``clean_prompts.py`` / ``prompt_specs.py``. This module
implements *separate* transformations for robustness experiments so that:

- We never silently edit clean templates for a single trial.
- Perturbation type is logged explicitly in results.

Planned families (implement when running Study B):

1. **popup_extra_screen** — Inject an extra permission dialog, onboarding page,
   or confirmation before/after a critical step (same task seed, scripted UI).

2. **stale_action_step** — For action-granularity prompts only, replace one or
   two UI labels or ordering hints so that literal following fails unless the
   model recovers (tests brittle instruction-following).

3. **delay_timing** (optional) — Artificial delay before a key UI element
   appears (flaky timing); use sparingly.

The functions below are stubs: they return the goal unchanged until you wire
env-level hooks or prompt post-processing for each task.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PerturbationKind(str, Enum):
    NONE = "none"
    POPUP_EXTRA_SCREEN = "popup_extra_screen"
    STALE_ACTION_STEP = "stale_action_step"


@dataclass(frozen=True)
class PerturbationContext:
    task_class: str
    granularity: str
    seed: int
    kind: PerturbationKind


def apply_perturbation_to_goal(
    goal_text: str,
    ctx: PerturbationContext,
) -> tuple[str, dict[str, Any]]:
    """Return (possibly modified goal, metadata dict for logging).

    Clean (`none`) trials pass through unchanged.
    Any other perturbation must be explicitly implemented before use.
    """

    if ctx.kind is not PerturbationKind.NONE:
        raise NotImplementedError(
            f"Perturbation {ctx.kind.value!r} is declared but not implemented yet. "
            "Run clean trials with --perturbation none until Study B hooks are wired."
        )
    return goal_text, {"perturbation_applied": False, "kind": ctx.kind.value}


def describe_stale_action_protocol() -> str:
    """Human-readable protocol for stale-action experiments (for README)."""

    return (
        "For action-level prompts only, optionally swap one or two UI labels "
        "in the appended action block to outdated labels that do not match the "
        "current build, while leaving workflow/intent untouched for the same "
        "trial seed. Compare recovery vs. blind following."
    )
