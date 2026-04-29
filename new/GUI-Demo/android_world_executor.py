"""Bridge Doubao GUI actions to AndroidWorld JSONAction execution."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from action_parser import ParsedAction
from screenshot_utils import save_state_screenshot


class AndroidWorldExecutor:
    """Execute parsed Doubao actions through AndroidWorld's public env API."""

    def __init__(self, env: Any, *, transition_pause: float = 1.0) -> None:
        self.env = env
        self.transition_pause = transition_pause

    def current_state(self, *, wait_to_stabilize: bool = False) -> Any:
        return self.env.get_state(wait_to_stabilize=wait_to_stabilize)

    def save_current_screenshot(self, path: str | Path, *, wait_to_stabilize: bool = False) -> Path:
        state = self.current_state(wait_to_stabilize=wait_to_stabilize)
        return save_state_screenshot(state, path)

    def execute(
        self,
        parsed: ParsedAction,
        *,
        screen_size: tuple[int, int],
        before_state: Any | None = None,
    ) -> tuple[Any, bool]:
        """Execute one action and return `(json_action, done)`."""
        json_action = self.to_json_action(
            parsed, screen_size=screen_size, before_state=before_state
        )
        done = json_action.action_type in ("status", "answer")
        if parsed.action_name == "type" and _needs_clipboard_input(parsed.content or ""):
            self._execute_clipboard_input(
                parsed,
                screen_size=screen_size,
                before_state=before_state,
            )
            if self.transition_pause:
                time.sleep(self.transition_pause)
            return json_action, done
        self.env.execute_action(json_action)
        if self.transition_pause:
            time.sleep(self.transition_pause)
        return json_action, done

    def _execute_clipboard_input(
        self,
        parsed: ParsedAction,
        *,
        screen_size: tuple[int, int],
        before_state: Any | None = None,
    ) -> None:
        """Paste text through Clipper for characters adb input_text mangles."""
        from android_world.env import adb_utils  # pylint: disable=import-outside-toplevel
        from android_world.env import json_action as json_action_lib  # pylint: disable=import-outside-toplevel

        text = parsed.content or ""
        focus_point = _scaled_point(parsed.point, screen_size) if parsed.point is not None else _focused_editable_point(before_state)
        if focus_point is not None:
            x, y = focus_point
            self.env.execute_action(json_action_lib.JSONAction(action_type="click", x=x, y=y))
            time.sleep(0.5)

        adb_utils.launch_app("clipper", self.env.controller)
        time.sleep(0.5)
        adb_utils.issue_generic_request(
            [
                "shell",
                "am",
                "broadcast",
                "-a",
                "clipper.set",
                "-e",
                "text",
                _adb_shell_quote(text),
            ],
            self.env.controller,
        )
        adb_utils.press_back_button(self.env.controller)
        time.sleep(0.5)

        if focus_point is not None:
            x, y = focus_point
            self.env.execute_action(json_action_lib.JSONAction(action_type="click", x=x, y=y))
            time.sleep(0.5)

        adb_utils.issue_generic_request(
            [
                "shell",
                "input",
                "keycombination",
                "113",
                "29",
                "&&",
                "input",
                "keyevent",
                "67",
            ],
            self.env.controller,
        )
        time.sleep(0.5)
        adb_utils.issue_generic_request(
            ["shell", "input", "keyevent", "279"],
            self.env.controller,
        )

    def to_json_action(
        self,
        parsed: ParsedAction,
        *,
        screen_size: tuple[int, int],
        before_state: Any | None = None,
    ) -> Any:
        """Convert a parsed Doubao action to AndroidWorld JSONAction."""
        from android_world.env import json_action as json_action_lib

        name = parsed.action_name
        if name == "click":
            x, y = _scaled_point(parsed.point, screen_size)
            return json_action_lib.JSONAction(action_type="click", x=x, y=y)
        if name == "long_press":
            x, y = _scaled_point(parsed.point, screen_size)
            return json_action_lib.JSONAction(action_type="long_press", x=x, y=y)
        if name == "type":
            # Doubao GUI often uses type() to mean "set this field to ...".
            # AndroidWorld supports clear_text, which is more reliable than
            # asking the model to manually press backspace on the soft keyboard.
            if parsed.point is not None:
                x, y = _scaled_point(parsed.point, screen_size)
                return json_action_lib.JSONAction(
                    action_type="input_text",
                    x=x,
                    y=y,
                    text=parsed.content or "",
                    clear_text=bool(parsed.content),
                )
            return json_action_lib.JSONAction(
                action_type="input_text",
                text=parsed.content or "",
                clear_text=bool(parsed.content),
            )
        if name == "scroll":
            # AndroidWorld's scroll uses UI-element `index` for anchoring; x/y on
            # JSONAction is ignored. When Doubao gives a point, look up the
            # smallest scrollable container containing that point and pass its
            # index. Otherwise fall back to a full-screen scroll.
            direction = _direction(parsed.direction)
            scroll_index = None
            if parsed.point is not None and before_state is not None:
                x, y = _scaled_point(parsed.point, screen_size)
                scroll_index = _find_scrollable_index(before_state, x, y)
            if scroll_index is not None:
                return json_action_lib.JSONAction(
                    action_type="scroll", index=scroll_index, direction=direction
                )
            return json_action_lib.JSONAction(action_type="scroll", direction=direction)
        if name == "open_app":
            app_name = _normalize_app_name(parsed.app_name or "")
            return json_action_lib.JSONAction(action_type="open_app", app_name=app_name)
        if name == "press_back":
            return json_action_lib.JSONAction(action_type="navigate_back")
        if name == "press_home":
            return json_action_lib.JSONAction(action_type="navigate_home")
        if name == "wait":
            return json_action_lib.JSONAction(action_type="wait")
        if name == "finished":
            return json_action_lib.JSONAction(action_type="status", goal_status="complete")
        if name == "answer":
            return json_action_lib.JSONAction(action_type="answer", text=parsed.content or "")
        if name == "drag":
            # AndroidWorld's JSONAction dataclass does not expose drag coordinates
            # in this checked-in version, so use a swipe in the nearest direction.
            return json_action_lib.JSONAction(
                action_type="swipe",
                direction=_drag_direction(parsed.start_point, parsed.end_point),
            )
        raise ValueError(f"Unsupported Doubao GUI action: {parsed.action_text}")


def _scaled_point(point: tuple[float, float] | None, screen_size: tuple[int, int]) -> tuple[int, int]:
    if point is None:
        raise ValueError("Action requires point='<point>x y</point>'.")
    width, height = screen_size
    x = int(round(point[0] / 1000.0 * width))
    y = int(round(point[1] / 1000.0 * height))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _needs_clipboard_input(text: str) -> bool:
    return any(char in text for char in ("'", '"', "\\", "`", "|", "&", "<", ">", "\n"))


def _adb_shell_quote(text: str) -> str:
    """Quote one argument for Android's `/system/bin/sh`."""
    return "'" + text.replace("'", "'\\''") + "'"


def _focused_editable_point(state: Any | None) -> tuple[int, int] | None:
    elements = getattr(state, "ui_elements", None) or []
    candidates: list[tuple[int, int, int]] = []
    for element in elements:
        if not getattr(element, "is_editable", False):
            continue
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            continue
        try:
            x = int(round((float(bbox.x_min) + float(bbox.x_max)) / 2))
            y = int(round((float(bbox.y_min) + float(bbox.y_max)) / 2))
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        focused_rank = 0 if getattr(element, "is_focused", False) else 1
        candidates.append((focused_rank, -y, x))
    if not candidates:
        return None
    focused_rank, neg_y, x = sorted(candidates)[0]
    return x, -neg_y


def _direction(direction: str | None) -> str:
    value = (direction or "").strip().lower()
    if value not in {"up", "down", "left", "right"}:
        raise ValueError(f"Invalid scroll direction: {direction}")
    return value


def _find_scrollable_index(state: Any, x: int, y: int) -> int | None:
    """Return the index of the smallest scrollable UI element containing (x, y).

    Falls back to the smallest element containing (x, y) when no scrollable
    container is a hit; AndroidWorld's scroll uses that element's bbox as the
    anchor, which is still better than a full-screen fallback on dense lists.
    """
    elements = getattr(state, "ui_elements", None) or []
    best_scrollable: tuple[int, int] | None = None  # (area, index)
    best_any: tuple[int, int] | None = None
    for idx, element in enumerate(elements):
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            continue
        try:
            x_min = int(bbox.x_min)
            y_min = int(bbox.y_min)
            x_max = int(bbox.x_max)
            y_max = int(bbox.y_max)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        area = (x_max - x_min) * (y_max - y_min)
        if getattr(element, "is_scrollable", False):
            if best_scrollable is None or area < best_scrollable[0]:
                best_scrollable = (area, idx)
        if best_any is None or area < best_any[0]:
            best_any = (area, idx)
    if best_scrollable is not None:
        return best_scrollable[1]
    if best_any is not None:
        return best_any[1]
    return None


def _drag_direction(
    start: tuple[float, float] | None,
    end: tuple[float, float] | None,
) -> str:
    if start is None or end is None:
        raise ValueError("drag requires start_point and end_point.")
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _normalize_app_name(app_name: str) -> str:
    value = (app_name or "").strip()
    if not value:
        return value
    alias_map = {
        "文件": "Files",
        "文件管理器": "Files",
        "files by google": "Files by Google",
        "设置": "Settings",
        "日历": "Calendar",
        "联系人": "Contacts",
        "信息": "Messages",
        "相机": "Camera",
    }
    lowered = value.lower()
    return alias_map.get(lowered, alias_map.get(value, value))
