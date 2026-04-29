"""Prompt/message construction for Doubao phone GUI control.

This matches the official Volc Doubao GUI demo pattern:
  1. one system-style prompt as role='user'
  2. alternating (user image_url, assistant output) pairs (sliding window)
  3. the current screenshot as the last user message
Optional Android accessibility UI text is attached ONLY to the current message
(never inside history turns), so history stays close to the model's training
distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from prompt import PHONE_USE_DOUBAO
from screenshot_utils import image_path_to_data_url


@dataclass(frozen=True)
class HistoryTurn:
    """One past (user_image, assistant_output) pair."""

    screenshot_path: str
    assistant_output: str
    step: int = 0


def build_step_messages(
    *,
    goal: str,
    task_params: dict[str, Any] | None = None,
    screenshot_path: str,
    history_turns: list[HistoryTurn] | None = None,
    max_screenshot_history: int = 5,
    max_text_history_chars: int = 6000,
    current_ui_text: str = "",
    language: str = "Chinese",
) -> list[dict[str, Any]]:
    """Build an official-style Doubao GUI message list for the next action."""
    turns = list(history_turns or [])
    system_text = _system_prompt(goal=goal, task_params=task_params, language=language)
    messages: list[dict[str, Any]] = [{"role": "user", "content": system_text}]

    keep_from = max(0, len(turns) - max(0, max_screenshot_history - 1))
    older_turns = turns[:keep_from]
    if older_turns:
        messages.append(_text_history_message(older_turns, max_chars=max_text_history_chars))
    for turn in turns[keep_from:]:
        messages.append(_image_only_user_message(turn.screenshot_path))
        messages.append({"role": "assistant", "content": turn.assistant_output})

    messages.append(_current_user_message(screenshot_path, current_ui_text))
    return messages


def _system_prompt(
    *,
    goal: str,
    task_params: dict[str, Any] | None,
    language: str,
) -> str:
    base = PHONE_USE_DOUBAO.format(instruction=goal.strip(), language=language)
    params_text = ""
    if task_params:
        params_text = (
            "\n## Canonical AndroidWorld task parameters\n"
            "These values come from the official task object and are the values used by "
            "the AndroidWorld success checker. If the natural-language instruction adds "
            "formatting punctuation around a parameter, type the canonical parameter value "
            "exactly instead of duplicating that surrounding punctuation.\n"
            f"{json.dumps(task_params, ensure_ascii=False, indent=2, default=str)}\n"
        )
    extras = (
        params_text
        +
        "\n## Additional Rules (AndroidWorld bridge)\n"
        "- For every step, ALWAYS output two lines in this order: "
        "Thought: <brief reasoning> and Action: <one valid action call>.\n"
        "- The <point>x y</point> coordinates are normalized 0..1000 relative to the screenshot.\n"
        "- Do not treat a truncated filename prefix as proof of identity; confirm the full target name.\n"
        "- If the target is not visible, prefer navigate/search/scroll/wait over guessing.\n"
        "- For text input, prefer: click to focus, then type(content='...'). "
        "type(content='...') replaces the current focused field with the full desired value; "
        "do not manually press soft-keyboard backspace to edit text.\n"
        "- Preserve required punctuation exactly when typing names or values. "
        "For a filename extension field that visibly includes a dot or has a hint like '.md', "
        "type the full dotted extension such as '.txt', not just 'txt'.\n"
        "- Allowed actions also include: press_back(), press_home(), wait(), open_app(app_name='...').\n"
        "- For open_app(app_name='...'), use the canonical app label in English "
        "(e.g., 'Files', 'Settings', 'Calendar', 'Contacts', 'Messages').\n"
        "\n## When to declare completion\n"
        "After each action, check whether the goal-visible condition is met. If yes, "
        "output Action: finished(content='complete') (or answer(content='...') for QA) "
        "and stop acting. Do not keep navigating after success. Completion checks by task family:\n"
        "- Delete / remove: the target is no longer present after appropriate search, scrolling, or navigation; absence from only the current viewport is not enough for long lists.\n"
        "- Create / edit / save: the expected file/note/entry is visible with the required content.\n"
        "- Move / copy: the item is visible in the destination folder.\n"
        "- Toggle / settings / Wi-Fi / Bluetooth / DND: the setting shows the desired state.\n"
        "- Question-answering (goal contains 'what', 'how many', 'answer', '?'): use "
        "answer(content='...') with the final answer string.\n"
        "- Timer / stopwatch: the timer shows the requested state (running / paused / reset).\n"
        "If you are unsure, prefer one more verification step (e.g. scroll or wait) over a false finish.\n"
    )
    return base + extras


def _image_only_user_message(screenshot_path: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(screenshot_path)},
            }
        ],
    }


def _text_history_message(turns: list[HistoryTurn], *, max_chars: int) -> dict[str, Any]:
    lines = ["Older action history without screenshots:"]
    for index, turn in enumerate(turns, start=1):
        step_label = turn.step or index
        lines.append(f"Step {step_label}: {_compact_assistant_output(turn.assistant_output)}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[-max_chars:].lstrip()
        text = "Older action history truncated to recent text:\n" + text
    return {"role": "user", "content": text}


def _compact_assistant_output(output: str, *, max_chars: int = 600) -> str:
    text = (output or "").strip()
    if not text:
        return "(empty)"
    if len(text) > max_chars:
        return text[: max_chars - 15].rstrip() + "...[truncated]"
    return text


def _current_user_message(screenshot_path: str, current_ui_text: str) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    ui = _sanitize_ui_text(current_ui_text)
    if ui:
        parts.append(
            {
                "type": "text",
                "text": (
                    "Accessibility text of the current screen (for disambiguation only; "
                    "still output the action using <point>x y</point> coordinates):\n"
                    f"{ui}"
                ),
            }
        )
    parts.append(
        {
            "type": "image_url",
            "image_url": {"url": image_path_to_data_url(screenshot_path)},
        }
    )
    return {"role": "user", "content": parts}


def _sanitize_ui_text(current_ui_text: str) -> str:
    text = (current_ui_text or "").strip()
    if not text:
        return ""
    if len(text) > 5000:
        return text[:5000] + "\n...[truncated]"
    return text
