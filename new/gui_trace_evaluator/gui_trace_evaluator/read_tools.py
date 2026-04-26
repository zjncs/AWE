"""Read-only fallback tools for post-execution GUI state verification."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from gui_trace_evaluator.record_adapter import NormalizedRecord


DEFAULT_ALLOWED_ROOTS = (
    "/sdcard",
    "/storage/emulated/0",
)
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


@dataclass(frozen=True)
class ReadToolConfig:
    enabled: bool = True
    adb_path: str | None = None
    adb_serial: str | None = None
    timeout_seconds: int = 10
    max_output_chars: int = 6000
    allowed_roots: tuple[str, ...] = DEFAULT_ALLOWED_ROOTS


class ReadToolRunner:
    """Execute a small allowlisted set of read-only Android state tools."""

    def __init__(self, config: ReadToolConfig | None = None) -> None:
        self.config = config or ReadToolConfig()

    def run_requests(
        self,
        record: NormalizedRecord,
        checkpoint: dict[str, Any],
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return [
                {
                    "type": "read_tool_result",
                    "status": "skipped",
                    "error": "Read tools are disabled.",
                    "request": request,
                }
                for request in requests
            ]
        results = []
        for index, request in enumerate(requests, start=1):
            results.append(self._run_one(record, checkpoint, request, index=index))
        return results

    def _run_one(
        self,
        record: NormalizedRecord,
        checkpoint: dict[str, Any],
        request: dict[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        tool = str(request.get("tool") or "").strip()
        try:
            if tool == "list_dir":
                path = _request_path(request)
                return self._adb_shell_result(
                    request,
                    index=index,
                    command=f"ls -la {_quote_allowed_path(path, self.config.allowed_roots)}",
                )
            if tool == "stat_path":
                path = _request_path(request)
                quoted_path = _quote_allowed_path(path, self.config.allowed_roots)
                result = self._adb_shell_result(
                    request,
                    index=index,
                    command=(
                        f"if [ -e {quoted_path} ]; then "
                        f"echo __AWE_PATH_EXISTS=true__; stat {quoted_path}; "
                        "else echo __AWE_PATH_EXISTS=false__; fi"
                    ),
                )
                _annotate_stat_result(result)
                return result
            if tool == "find_file":
                root = str(request.get("root") or request.get("path") or "/sdcard").strip()
                name = str(request.get("name") or request.get("target") or "").strip()
                if not name:
                    raise ValueError("find_file requires name.")
                quoted_root = _quote_allowed_path(root, self.config.allowed_roots)
                quoted_name = shlex.quote(name)
                result = self._adb_shell_result(
                    request,
                    index=index,
                    command=f"find {quoted_root} -name {quoted_name} 2>/dev/null | head -50",
                )
                _annotate_find_result(result)
                return result
            if tool == "read_text_file":
                path = _request_path(request)
                return self._adb_shell_result(
                    request,
                    index=index,
                    command=f"sed -n '1,200p' {_quote_allowed_path(path, self.config.allowed_roots)}",
                )
            return {
                "type": "read_tool_result",
                "index": index,
                "tool": tool or "unknown",
                "status": "rejected",
                "request": request,
                "error": "Unsupported read-only tool.",
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {
                "type": "read_tool_result",
                "index": index,
                "tool": tool or "unknown",
                "status": "error",
                "request": request,
                "error": str(exc),
            }

    def _adb_shell_result(
        self,
        request: dict[str, Any],
        *,
        index: int,
        command: str,
    ) -> dict[str, Any]:
        adb = self._adb_path()
        adb_command = [adb]
        if self.config.adb_serial:
            adb_command.extend(["-s", self.config.adb_serial])
        adb_command.extend(["shell", command])
        try:
            completed = subprocess.run(
                adb_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.config.timeout_seconds,
                check=False,
            )
            output = _trim_output(completed.stdout or "", self.config.max_output_chars)
            return {
                "type": "read_tool_result",
                "index": index,
                "tool": request.get("tool"),
                "status": "ok" if completed.returncode == 0 else "command_error",
                "request": request,
                "command": " ".join(shlex.quote(part) for part in adb_command),
                "returncode": completed.returncode,
                "output": output,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "type": "read_tool_result",
                "index": index,
                "tool": request.get("tool"),
                "status": "timeout",
                "request": request,
                "command": " ".join(shlex.quote(part) for part in adb_command),
                "error": str(exc),
            }

    def _adb_path(self) -> str:
        candidates = [
            self.config.adb_path,
            os.environ.get("ADB_PATH"),
            os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
            "adb",
        ]
        for candidate in candidates:
            if candidate and (candidate == "adb" or os.path.exists(candidate)):
                return candidate
        raise FileNotFoundError("adb not found. Set --adb_path or ADB_PATH.")


def default_read_requests(record: NormalizedRecord, checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    """Build conservative final-state evidence requests when the model asks for fallback but omits tools."""
    task_goal = f"{record.task}\n{record.goal}\n{checkpoint.get('description', '')}"
    if not _looks_like_file_task(task_goal):
        return []
    requests: list[dict[str, Any]] = []
    dirs = _mentioned_android_dirs(task_goal)
    filenames = _mentioned_filenames(task_goal)
    for directory in dirs:
        requests.append(
            {
                "tool": "list_dir",
                "path": f"/sdcard/{directory}",
                "reason": "Collect final directory listing for a file-related checkpoint.",
                "source": "default_file_task_policy",
            }
        )
    for filename in filenames:
        requests.append(
            {
                "tool": "find_file",
                "root": "/sdcard",
                "name": filename,
                "reason": "Locate the target file after execution.",
                "source": "default_file_task_policy",
            }
        )
        for directory in dirs:
            requests.append(
                {
                    "tool": "stat_path",
                    "path": f"/sdcard/{directory}/{filename}",
                    "reason": "Check exact target path existence after execution.",
                    "source": "default_file_task_policy",
                }
            )
    return _dedupe_requests(requests)


def _looks_like_file_task(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("file", "folder", "directory", "note", "move", "delete", "copy"))


def _mentioned_android_dirs(text: str) -> list[str]:
    found = [directory for directory in COMMON_ANDROID_DIRS if re.search(rf"\b{re.escape(directory)}\b", text)]
    return list(dict.fromkeys(found))


def _mentioned_filenames(text: str) -> list[str]:
    matches = re.findall(r"\b[\w.-]+\.[A-Za-z0-9]{1,8}\b", text)
    return list(dict.fromkeys(matches))


def _dedupe_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for request in requests:
        key = (
            request.get("tool"),
            request.get("path"),
            request.get("root"),
            request.get("name"),
        )
        if key not in seen:
            deduped.append(request)
            seen.add(key)
    return deduped


def _request_path(request: dict[str, Any]) -> str:
    path = str(request.get("path") or request.get("target") or "").strip()
    if not path:
        raise ValueError(f"{request.get('tool')} requires path.")
    return path


def _quote_allowed_path(path: str, allowed_roots: tuple[str, ...]) -> str:
    normalized = _normalize_android_path(path)
    if not any(normalized == root or normalized.startswith(f"{root}/") for root in allowed_roots):
        raise ValueError(f"Path is outside allowed read roots: {normalized}")
    return shlex.quote(normalized)


def _normalize_android_path(path: str) -> str:
    if not path.startswith("/"):
        raise ValueError(f"Only absolute Android paths are allowed: {path}")
    return "/" + str(PurePosixPath(path)).lstrip("/")


def _trim_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n...[truncated {len(output) - max_chars} chars]"


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


def _annotate_find_result(result: dict[str, Any]) -> None:
    output = str(result.get("output") or "")
    matches = [
        line.strip()
        for line in output.splitlines()
        if _looks_like_android_path_line(line.strip())
    ]
    result["matches"] = matches
    result["found"] = bool(matches)
    result["interpretation"] = (
        "Matching files were found." if matches else "No matching files were found."
    )


def _looks_like_android_path_line(line: str) -> bool:
    if not line:
        return False
    # adb/find output can be polluted by gRPC logs like:
    # I0425 ... ev_poll_posix.cc ...
    # Treat only absolute POSIX paths as valid find matches.
    if not line.startswith("/"):
        return False
    return True
