"""Record serialization for GUI-Demo AndroidWorld executions."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


COMMON_ANDROID_DIRS = (
    "Alarms",
    "DCIM",
    "Documents",
    "Download",
    "Downloads",
    "Movies",
    "Music",
    "Notifications",
    "Pictures",
    "Ringtones",
)

UI_ELEMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("idx", "index"),
    ("text", "text"),
    ("desc", "content_description"),
    ("hint", "hint_text"),
    ("class", "class_name"),
    ("enabled", "is_enabled"),
    ("visible", "is_visible"),
    ("clickable", "is_clickable"),
    ("long_clickable", "is_long_clickable"),
    ("editable", "is_editable"),
    ("checkable", "is_checkable"),
    ("checked", "is_checked"),
    ("selected", "is_selected"),
    ("focusable", "is_focusable"),
    ("focused", "is_focused"),
    ("scrollable", "is_scrollable"),
    ("resource", "resource_name"),
    ("package", "package_name"),
    ("bbox", "bbox_pixels"),
    ("center", "bbox_center"),
)

UI_COMPACT_FIELDS = (
    "idx",
    "text",
    "desc",
    "hint",
    "class",
    "enabled",
    "visible",
    "clickable",
    "editable",
    "scrollable",
    "resource",
    "bbox",
    "center",
)


def step_record(
    *,
    step: int,
    raw_response: str,
    parsed_action: Any,
    json_action: Any,
    before_screenshot_path: str,
    after_screenshot_path: str,
    before_state: Any,
    after_state: Any,
    summary: str,
) -> dict[str, Any]:
    """Build one evaluator-compatible trace step."""
    action_json = _json_action_dict(json_action)
    action_target = _action_target(parsed_action, action_json, before_state)
    return {
        "step": step,
        "thinking": parsed_action.thought,
        "thought": parsed_action.thought,
        "action": parsed_action.action_text,
        "action_json": action_json,
        "action_target": action_target,
        "action_target_ui": _action_target_to_text(action_target),
        "summary": summary,
        "raw_response": raw_response,
        "before_screenshot_path": before_screenshot_path,
        "after_screenshot_path": after_screenshot_path,
        "before_ui": ui_to_text(before_state),
        "after_ui": ui_to_text(after_state),
    }


def build_record(
    *,
    task_name: str,
    goal: str,
    task_params: dict[str, Any] | None = None,
    seed: int,
    steps: list[dict[str, Any]],
    reward: float | int | None,
    success: bool | None,
    agent_done: bool,
    abort_reason: str | None,
    elapsed_seconds: float,
    post_execution_evidence: list[dict[str, Any]],
    model: str,
    llm_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one top-level record consumed by gui_trace_evaluator."""
    return {
        "task": task_name,
        "task_name": task_name,
        "base_goal": goal,
        "goal": goal,
        "goal_used": goal,
        "task_params": _json_safe(task_params or {}),
        "granularity": "intent",
        "seed": seed,
        "model": model,
        "reward": reward,
        "success": success,
        "agent_done": agent_done,
        "abort_reason": abort_reason,
        # Top-level "steps" is the step count; full list is trace["steps"].
        "step_count": len(steps),
        "steps": len(steps),
        "elapsed_seconds": elapsed_seconds,
        "trace": {
            "format_version": "gui_demo_android_world_v1",
            "steps": steps,
        },
        "post_execution_evidence": post_execution_evidence,
        "llm_usage": llm_usage or {},
    }


def write_records(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def ui_to_text(state: Any, *, max_elements: int = 120) -> str:
    """Serialize UI elements as compact key-value lines for trace inspection."""
    elements = getattr(state, "ui_elements", []) or []
    rows = [_ui_element_row(index, element) for index, element in enumerate(elements[:max_elements])]
    if not rows:
        return ""
    lines = [_format_ui_line(row) for row in rows]
    if len(elements) > max_elements:
        lines.append(f"... truncated {len(elements) - max_elements} UI elements")
    return "\n".join(lines)


def _ui_element_row(index: int, element: Any) -> dict[str, str]:
    row: dict[str, str] = {}
    for name, attr in UI_ELEMENT_FIELDS:
        if attr == "index":
            value = index
        elif attr == "bbox_center":
            value = _bbox_center(getattr(element, "bbox_pixels", None))
        elif attr == "bbox_pixels":
            value = _bbox_text(getattr(element, "bbox_pixels", None))
        else:
            value = getattr(element, attr, None)
        row[name] = _cell_text(value)
    return row


def _bbox_text(bbox: Any | None) -> str:
    if bbox is None:
        return ""
    return f"{int(bbox.x_min)},{int(bbox.y_min)},{int(bbox.x_max)},{int(bbox.y_max)}"


def _bbox_center(bbox: Any | None) -> str:
    if bbox is None:
        return ""
    x = (float(bbox.x_min) + float(bbox.x_max)) / 2
    y = (float(bbox.y_min) + float(bbox.y_max)) / 2
    return f"{int(round(x))},{int(round(y))}"


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "T" if value else "F"
    text = str(value).replace("\n", "\\n").strip()
    if len(text) > 80:
        text = text[:77].rstrip() + "..."
    return text


def _format_ui_line(row: dict[str, str], *, fields: tuple[str, ...] = UI_COMPACT_FIELDS) -> str:
    parts = []
    for field in fields:
        value = row.get(field, "")
        if field not in {"idx", "enabled", "visible", "clickable", "editable", "scrollable"} and not value:
            continue
        parts.append(f"{field}={json.dumps(value, ensure_ascii=False)}")
    return " ".join(parts)


def _action_target(parsed_action: Any, action_json: dict[str, Any], before_state: Any) -> dict[str, Any]:
    action_type = str(action_json.get("action_type") or parsed_action.action_name or "")
    target: dict[str, Any] = {
        "action_type": action_type,
        "raw_action": getattr(parsed_action, "action_text", ""),
        "is_pointer_action": action_type in {"click", "long_press", "double_tap", "input_text"},
        "point": None,
        "hit": False,
        "ui_index": None,
        "ui": None,
    }

    x = action_json.get("x")
    y = action_json.get("y")
    if x is not None and y is not None:
        try:
            point = (int(round(float(x))), int(round(float(y))))
        except (TypeError, ValueError):
            point = None
        if point is not None:
            target["point"] = {"x": point[0], "y": point[1]}
            match = _find_smallest_element_at(before_state, point[0], point[1])
            if match is not None:
                index, element = match
                row = _ui_element_row(index, element)
                target.update(
                    {
                        "hit": True,
                        "ui_index": index,
                        "ui": row,
                        "enabled": getattr(element, "is_enabled", None),
                        "clickable": getattr(element, "is_clickable", None),
                        "editable": getattr(element, "is_editable", None),
                        "scrollable": getattr(element, "is_scrollable", None),
                    }
                )
            return target

    index = action_json.get("index")
    if index is not None:
        try:
            element = (getattr(before_state, "ui_elements", []) or [])[int(index)]
        except (TypeError, ValueError, IndexError):
            return target
        row = _ui_element_row(int(index), element)
        target.update(
            {
                "hit": True,
                "ui_index": int(index),
                "ui": row,
                "enabled": getattr(element, "is_enabled", None),
                "clickable": getattr(element, "is_clickable", None),
                "editable": getattr(element, "is_editable", None),
                "scrollable": getattr(element, "is_scrollable", None),
            }
        )
    return target


def _find_smallest_element_at(state: Any, x: int, y: int) -> tuple[int, Any] | None:
    elements = getattr(state, "ui_elements", []) or []
    best: tuple[int, int, Any] | None = None
    for index, element in enumerate(elements):
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
        if best is None or area < best[0]:
            best = (area, index, element)
    if best is None:
        return None
    return best[1], best[2]


def _action_target_to_text(action_target: dict[str, Any]) -> str:
    ui = action_target.get("ui")
    if not isinstance(ui, dict):
        point = action_target.get("point") or {}
        if point:
            return (
                f"action_type={action_target.get('action_type')} "
                f"point=({point.get('x')},{point.get('y')}) hit=False"
            )
        return f"action_type={action_target.get('action_type')} hit=False"

    columns = [
        "action_type",
        "point",
        "hit",
        "idx",
        "text",
        "desc",
        "class",
        "enabled",
        "clickable",
        "editable",
        "scrollable",
        "bbox",
        "center",
    ]
    point = action_target.get("point") or {}
    row = {
        "action_type": _cell_text(action_target.get("action_type")),
        "point": _cell_text(f"{point.get('x')},{point.get('y')}" if point else ""),
        "hit": _cell_text(action_target.get("hit")),
        "idx": _cell_text(ui.get("idx")),
        "text": _cell_text(ui.get("text")),
        "desc": _cell_text(ui.get("desc")),
        "class": _cell_text(ui.get("class")),
        "enabled": _cell_text(ui.get("enabled")),
        "clickable": _cell_text(ui.get("clickable")),
        "editable": _cell_text(ui.get("editable")),
        "scrollable": _cell_text(ui.get("scrollable")),
        "bbox": _cell_text(ui.get("bbox")),
        "center": _cell_text(ui.get("center")),
    }
    return _format_ui_line(row, fields=tuple(columns))


def action_summary(parsed_action: Any, json_action: Any) -> str:
    details = _json_action_dict(json_action)
    return f"Executed {parsed_action.action_name}: {json.dumps(details, ensure_ascii=False)}"


def collect_post_execution_evidence(
    *,
    goal: str,
    task_name: str,
    adb_path: str,
    console_port: int,
    final_state: Any | None = None,
    final_screenshot_path: str | None = None,
) -> list[dict[str, Any]]:
    """Collect conservative read-only evidence for post-execution final states."""
    evidence: list[dict[str, Any]] = []
    if final_state is not None or final_screenshot_path:
        final_ui = ui_to_text(final_state) if final_state is not None else ""
        evidence.append(
            {
                "type": "final_state_snapshot",
                "tool": "final_state_snapshot",
                "status": "ok",
                "request": {
                    "tool": "final_state_snapshot",
                    "source": "gui_demo_post_execution",
                },
                "screenshot_path": final_screenshot_path or "",
                "final_ui": final_ui,
                "output": _final_state_output(
                    final_ui=final_ui,
                    final_screenshot_path=final_screenshot_path or "",
                ),
                "source": "gui_demo_post_execution",
            }
        )

    if "Files" not in task_name and not _looks_like_file_goal(goal):
        return evidence
    dirs = _mentioned_android_dirs(goal)
    filenames = _mentioned_filenames(goal)
    for directory in dirs:
        path = f"/sdcard/{directory}"
        evidence.append(
            _adb_shell_evidence(
                adb_path=adb_path,
                console_port=console_port,
                tool="list_dir",
                request={"tool": "list_dir", "path": path, "source": "gui_demo_post_execution"},
                shell_command=f"ls -la {shlex.quote(path)}",
            )
        )
    for filename in filenames:
        find_result = _adb_shell_evidence(
            adb_path=adb_path,
            console_port=console_port,
            tool="find_file",
            request={
                "tool": "find_file",
                "root": "/sdcard",
                "name": filename,
                "source": "gui_demo_post_execution",
            },
            shell_command=f"find /sdcard -name {shlex.quote(filename)} 2>/dev/null | head -50",
        )
        matches = [
            line.strip()
            for line in str(find_result.get("output") or "").splitlines()
            if _looks_like_android_path(line.strip())
        ]
        find_result["matches"] = matches
        find_result["found"] = bool(matches)
        if not matches:
            find_result["status"] = "not_found"
            find_result["output"] = ""
        find_result["interpretation"] = (
            "Matching files were found." if matches else "No matching files were found."
        )
        evidence.append(find_result)
        for directory in dirs:
            path = f"/sdcard/{directory}/{filename}"
            evidence.append(
                _adb_shell_evidence(
                    adb_path=adb_path,
                    console_port=console_port,
                    tool="stat_path",
                    request={"tool": "stat_path", "path": path, "source": "gui_demo_post_execution"},
                    shell_command=(
                        f"if [ -e {shlex.quote(path)} ]; then "
                        f"echo __AWE_PATH_EXISTS=true__; stat {shlex.quote(path)}; "
                        "else echo __AWE_PATH_EXISTS=false__; fi"
                    ),
                )
            )
            _annotate_stat_result(evidence[-1])
    return evidence


def _final_state_output(*, final_ui: str, final_screenshot_path: str) -> str:
    parts = []
    if final_screenshot_path:
        parts.append(f"Final screenshot path: {final_screenshot_path}")
    if final_ui:
        parts.append(f"Final UI text:\n{final_ui}")
    if not parts:
        return "No final UI text or screenshot path was available."
    return "\n\n".join(parts)


def _json_action_dict(json_action: Any) -> dict[str, Any]:
    if hasattr(json_action, "as_dict"):
        return json_action.as_dict(skip_none=True)
    if isinstance(json_action, dict):
        return json_action
    return {"repr": repr(json_action)}


def _adb_shell_evidence(
    *,
    adb_path: str,
    console_port: int,
    tool: str,
    request: dict[str, Any],
    shell_command: str,
) -> dict[str, Any]:
    command = [adb_path, "-s", f"emulator-{console_port}", "shell", shell_command]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        output = completed.stdout or ""
        if len(output) > 6000:
            output = output[:6000] + f"\n...[truncated {len(output) - 6000} chars]"
        return {
            "type": "read_tool_result",
            "tool": tool,
            "status": "ok" if completed.returncode == 0 else "command_error",
            "request": request,
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": completed.returncode,
            "output": output,
            "source": "gui_demo_post_execution",
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {
            "type": "read_tool_result",
            "tool": tool,
            "status": "error",
            "request": request,
            "command": " ".join(shlex.quote(part) for part in command),
            "error": str(exc),
            "source": "gui_demo_post_execution",
        }


def _annotate_stat_result(result: dict[str, Any]) -> None:
    output = str(result.get("output") or "")
    if "__AWE_PATH_EXISTS=true__" in output:
        result["exists"] = True
        result["interpretation"] = "The requested path exists."
        result["output"] = output.replace("__AWE_PATH_EXISTS=true__\n", "").replace(
            "__AWE_PATH_EXISTS=true__", ""
        )
    elif "__AWE_PATH_EXISTS=false__" in output:
        result["status"] = "ok"
        result["exists"] = False
        result["interpretation"] = "The requested path does not exist."
        result["output"] = output.replace("__AWE_PATH_EXISTS=false__\n", "").replace(
            "__AWE_PATH_EXISTS=false__", ""
        )


def _looks_like_file_goal(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("file", "folder", "directory", "note", "move", "delete", "copy"))


def _mentioned_android_dirs(text: str) -> list[str]:
    found = [directory for directory in COMMON_ANDROID_DIRS if re.search(rf"\b{re.escape(directory)}\b", text)]
    return list(dict.fromkeys(found))


def _mentioned_filenames(text: str) -> list[str]:
    matches = re.findall(r"\b[\w.-]+\.[A-Za-z0-9]{1,8}\b", text)
    return list(dict.fromkeys(matches))


def _looks_like_android_path(line: str) -> bool:
    return line.startswith(("/sdcard/", "/storage/", "/mnt/sdcard/"))
