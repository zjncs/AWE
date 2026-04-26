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
    return {
        "step": step,
        "thinking": parsed_action.thought,
        "thought": parsed_action.thought,
        "action": parsed_action.action_text,
        "action_json": _json_action_dict(json_action),
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
    seed: int,
    steps: list[dict[str, Any]],
    reward: float | int | None,
    success: bool | None,
    agent_done: bool,
    abort_reason: str | None,
    elapsed_seconds: float,
    post_execution_evidence: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Build one top-level record consumed by gui_trace_evaluator."""
    return {
        "task": task_name,
        "task_name": task_name,
        "base_goal": goal,
        "goal": goal,
        "goal_used": goal,
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
    }


def write_records(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def ui_to_text(state: Any, *, max_elements: int = 80) -> str:
    """Serialize visible UI elements into compact text for trace inspection."""
    elements = getattr(state, "ui_elements", []) or []
    lines = []
    for index, element in enumerate(elements[:max_elements]):
        text = getattr(element, "text", None) or ""
        desc = getattr(element, "content_description", None) or ""
        cls = getattr(element, "class_name", None) or ""
        clickable = getattr(element, "is_clickable", None)
        editable = getattr(element, "is_editable", None)
        bbox = getattr(element, "bbox_pixels", None)
        bbox_text = ""
        if bbox is not None:
            bbox_text = f" bbox=({int(bbox.x_min)},{int(bbox.y_min)},{int(bbox.x_max)},{int(bbox.y_max)})"
        if text or desc or clickable or editable:
            lines.append(
                f"[{index}] text={text!r} desc={desc!r} class={cls!r} "
                f"clickable={clickable} editable={editable}{bbox_text}"
            )
    return "\n".join(lines)


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
        matches = [line.strip() for line in str(find_result.get("output") or "").splitlines() if line.strip()]
        find_result["matches"] = matches
        find_result["found"] = bool(matches)
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
