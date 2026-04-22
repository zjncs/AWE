"""Small baseline task probes for evaluator calibration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from awe.trace_serialization import TRACE_FORMAT_VERSION


BASELINE_PROBE_TASKS: dict[str, tuple[str, ...]] = {
    "simple": ("ContactsAddContact", "SimpleSmsSend"),
    "complex": (
        "SimpleCalendarAddOneEvent",
        "RecipeAddSingleRecipe",
        "ClockTimerEntry",
        "ExpenseAddSingle",
    ),
    "main_six": (
        "ClockTimerEntry",
        "SimpleCalendarAddOneEvent",
        "ExpenseAddSingle",
        "FilesMoveFile",
        "RecipeAddSingleRecipe",
        "SimpleSmsSendReceivedAddress",
    ),
}


def get_sample_probe_records() -> list[dict[str, Any]]:
    """Returns synthetic traces covering simple and complex baseline tasks."""
    return deepcopy(_SAMPLE_PROBE_RECORDS)


def preset_task_names(name: str) -> tuple[str, ...]:
    if name != "baseline_probe":
        raise ValueError(f"Unknown preset: {name}")
    return BASELINE_PROBE_TASKS["simple"] + BASELINE_PROBE_TASKS["complex"]


def _trace(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"format_version": TRACE_FORMAT_VERSION, "steps": steps}


_SAMPLE_PROBE_RECORDS: list[dict[str, Any]] = [
    {
        "study": "trace_evaluator_sample_probe",
        "task": "ContactsAddContact",
        "granularity": "intent",
        "base_goal": "Create a new contact for Alex Chen with phone number 555-0102.",
        "goal_used": "Create a new contact for Alex Chen with phone number 555-0102.",
        "reward": 1.0,
        "success": True,
        "agent_done": True,
        "steps": 5,
        "trace": _trace(
            [
                {
                    "step": 1,
                    "action": {"action_type": "open_app", "app_name": "Contacts"},
                    "reason": "Open the app needed to create a contact.",
                    "summary": "Contacts opened on the contact list.",
                },
                {
                    "step": 2,
                    "action": {"action_type": "click", "index": 4},
                    "reason": "Start a new contact.",
                    "summary": "New contact form is visible.",
                },
                {
                    "step": 3,
                    "action": {"action_type": "input_text", "index": 7, "text": "Alex Chen"},
                    "reason": "Enter the requested name.",
                    "summary": "The name field contains Alex Chen.",
                },
                {
                    "step": 4,
                    "action": {"action_type": "input_text", "index": 11, "text": "555-0102"},
                    "reason": "Enter the requested phone number.",
                    "summary": "The phone field contains 555-0102.",
                },
                {
                    "step": 5,
                    "action": {"action_type": "click", "index": 2},
                    "reason": "Save the contact.",
                    "summary": "The contact detail page for Alex Chen is shown after saving.",
                },
            ]
        ),
    },
    {
        "study": "trace_evaluator_sample_probe",
        "task": "SimpleSmsSend",
        "granularity": "intent",
        "base_goal": "Send the message 'bring tea' to 555-0199.",
        "goal_used": "Send the message 'bring tea' to 555-0199.",
        "reward": 0.0,
        "success": False,
        "agent_done": True,
        "steps": 4,
        "trace": _trace(
            [
                {
                    "step": 1,
                    "action": {"action_type": "open_app", "app_name": "Simple SMS Messenger"},
                    "reason": "Open SMS.",
                    "summary": "Simple SMS Messenger opened.",
                },
                {
                    "step": 2,
                    "action": {"action_type": "click", "index": 5},
                    "reason": "Compose a message.",
                    "summary": "Compose screen opened.",
                },
                {
                    "step": 3,
                    "action": {"action_type": "input_text", "index": 1, "text": "555-0198"},
                    "reason": "Enter recipient.",
                    "summary": "Recipient field contains 555-0198, not 555-0199.",
                },
                {
                    "step": 4,
                    "action": {"action_type": "input_text", "index": 6, "text": "bring tea"},
                    "reason": "Enter the body.",
                    "summary": "Message body contains bring tea, but no send action occurred.",
                },
            ]
        ),
    },
    {
        "study": "trace_evaluator_sample_probe",
        "task": "SimpleCalendarAddOneEvent",
        "granularity": "workflow",
        "base_goal": (
            "Create a calendar event named Project Review on May 6, 2026 at "
            "14:00 for 60 minutes with description 'slides and budget'."
        ),
        "goal_used": (
            "Create a calendar event named Project Review on May 6, 2026 at "
            "14:00 for 60 minutes with description 'slides and budget'."
        ),
        "reward": 1.0,
        "success": True,
        "agent_done": True,
        "steps": 8,
        "trace": _trace(
            [
                {
                    "step": 1,
                    "action": {"action_type": "open_app", "app_name": "Simple Calendar Pro"},
                    "reason": "Open calendar.",
                    "summary": "Simple Calendar Pro opened.",
                },
                {
                    "step": 2,
                    "action": {"action_type": "click", "index": 15},
                    "reason": "Select target date.",
                    "summary": "May 6, 2026 is selected.",
                },
                {
                    "step": 3,
                    "action": {"action_type": "click", "index": 3},
                    "reason": "Create a new event.",
                    "summary": "New event form opened for May 6, 2026.",
                },
                {
                    "step": 4,
                    "action": {"action_type": "input_text", "index": 8, "text": "Project Review"},
                    "reason": "Enter title.",
                    "summary": "Title is Project Review.",
                },
                {
                    "step": 5,
                    "action": {"action_type": "input_text", "index": 11, "text": "slides and budget"},
                    "reason": "Enter description.",
                    "summary": "Description is slides and budget.",
                },
                {
                    "step": 6,
                    "action": {"action_type": "click", "index": 12},
                    "reason": "Set start time.",
                    "summary": "Start time is 14:00.",
                },
                {
                    "step": 7,
                    "action": {"action_type": "click", "index": 13},
                    "reason": "Set duration.",
                    "summary": "End time is 15:00, making the event 60 minutes.",
                },
                {
                    "step": 8,
                    "action": {"action_type": "click", "index": 1},
                    "reason": "Save the event.",
                    "summary": "Event saved and appears on May 6, 2026.",
                },
            ]
        ),
    },
    {
        "study": "trace_evaluator_sample_probe",
        "task": "RecipeAddSingleRecipe",
        "granularity": "action",
        "base_goal": (
            "Create a recipe named Lemon Rice with 2 servings, 15 minutes prep "
            "time, ingredients rice and lemon, and directions 'mix and serve'."
        ),
        "goal_used": (
            "Create a recipe named Lemon Rice with 2 servings, 15 minutes prep "
            "time, ingredients rice and lemon, and directions 'mix and serve'."
        ),
        "reward": 0.0,
        "success": False,
        "agent_done": True,
        "steps": 6,
        "trace": _trace(
            [
                {
                    "step": 1,
                    "action": {"action_type": "open_app", "app_name": "Broccoli"},
                    "reason": "Open recipe app.",
                    "summary": "Broccoli opened.",
                },
                {
                    "step": 2,
                    "action": {"action_type": "click", "index": 2},
                    "reason": "Start new recipe.",
                    "summary": "New recipe form is visible.",
                },
                {
                    "step": 3,
                    "action": {"action_type": "input_text", "index": 4, "text": "Lemon Rice"},
                    "reason": "Enter recipe title.",
                    "summary": "Title field contains Lemon Rice.",
                },
                {
                    "step": 4,
                    "action": {"action_type": "input_text", "index": 6, "text": "2"},
                    "reason": "Enter servings.",
                    "summary": "Servings field contains 2.",
                },
                {
                    "step": 5,
                    "action": {"action_type": "input_text", "index": 9, "text": "rice; lemon"},
                    "reason": "Enter ingredients.",
                    "summary": "Ingredients field contains rice and lemon.",
                },
                {
                    "step": 6,
                    "action": {"action_type": "click", "index": 1},
                    "reason": "Save.",
                    "summary": "Recipe was saved without entering prep time or directions.",
                },
            ]
        ),
    },
]
