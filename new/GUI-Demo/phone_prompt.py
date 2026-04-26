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

from dataclasses import dataclass
from typing import Any

from prompt import PHONE_USE_DOUBAO
from screenshot_utils import image_path_to_data_url


@dataclass(frozen=True)
class HistoryTurn:
    """One past (user_image, assistant_output) pair."""

    screenshot_path: str
    assistant_output: str


def build_step_messages(
    *,
    goal: str,
    screenshot_path: str,
    history_turns: list[HistoryTurn] | None = None,
    max_screenshot_history: int = 5,
    current_ui_text: str = "",
    language: str = "Chinese",
) -> list[dict[str, Any]]:
    """Build an official-style Doubao GUI message list for the next action."""
    turns = list(history_turns or [])
    system_text = _system_prompt(goal=goal, language=language)
    messages: list[dict[str, Any]] = [{"role": "user", "content": system_text}]

    keep_from = max(0, len(turns) - max(0, max_screenshot_history - 1))
    for turn in turns[keep_from:]:
        messages.append(_image_only_user_message(turn.screenshot_path))
        messages.append({"role": "assistant", "content": turn.assistant_output})

    messages.append(_current_user_message(screenshot_path, current_ui_text))
    return messages


def _system_prompt(*, goal: str, language: str) -> str:
    base = PHONE_USE_DOUBAO.format(instruction=goal.strip(), language=language)
    extras = (
        "\n## Additional Rules (AndroidWorld bridge)\n"
        "- The <point>x y</point> coordinates are normalized 0..1000 relative to the screenshot.\n"
        "- Do not treat a truncated filename prefix as proof of identity; confirm the full target name.\n"
        "- If the target is not visible, prefer navigate/search/scroll/wait over guessing.\n"
        "- For text input, prefer: click to focus, then type(content='...').\n"
        "- Allowed actions also include: press_back(), press_home(), wait(), open_app(app_name='...').\n"
        "- For open_app(app_name='...'), use the canonical app label in English "
        "(e.g., 'Files', 'Settings', 'Calendar', 'Contacts', 'Messages').\n"
        "\n## When to declare completion\n"
        "After each action, check whether the goal-visible condition is met. If yes, "
        "output Action: finished(content='complete') (or answer(content='...') for QA) "
        "and stop acting. Do not keep navigating after success. Completion checks by task family:\n"
        "- Delete / remove: the target filename is no longer listed on the current screen.\n"
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
