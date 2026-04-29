"""Normalize GUI-agent records from Android World/AWE or generic traces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_THOUGHT_RE = re.compile(r"(?:Thought|Reason)\s*:\s*(.*?)(?:\n\s*Action\s*:|$)", re.I | re.S)
_ACTION_RE = re.compile(r"Action\s*:\s*(.*)", re.I | re.S)


@dataclass(frozen=True)
class NormalizedStep:
    step: int
    thinking: str
    action: str
    summary: str
    before_screenshot_path: str
    after_screenshot_path: str
    before_ui: str = ""
    after_ui: str = ""
    action_target_ui: str = ""

    @property
    def evidence_screenshot_path(self) -> str:
        return self.after_screenshot_path or self.before_screenshot_path

    @property
    def evidence_screenshot_kind(self) -> str:
        return "after" if self.after_screenshot_path else "before"


@dataclass(frozen=True)
class NormalizedRecord:
    task: str
    goal: str
    granularity: str | None
    official_success: bool | None
    official_reward: Any
    raw: dict[str, Any]
    steps: list[NormalizedStep]


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a list of records from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "results", "evaluations"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Expected a JSON list or an object with records/results/evaluations.")


def normalize_record(
    record: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    image_root: str | Path | None = None,
) -> NormalizedRecord:
    """Normalize a GUI execution record without importing Android World."""
    steps_raw = _extract_steps(record)
    base = Path(base_dir).resolve() if base_dir else None
    root = Path(image_root).resolve() if image_root else None
    steps = [
        _normalize_step(step_data, index=index, base_dir=base, image_root=root)
        for index, step_data in enumerate(steps_raw, start=1)
    ]
    return NormalizedRecord(
        task=str(
            record.get("task")
            or record.get("task_name")
            or record.get("task_template")
            or "unknown_task"
        ),
        goal=str(
            record.get("base_goal")
            or record.get("goal")
            or record.get("goal_used")
            or record.get("instruction")
            or ""
        ),
        granularity=record.get("granularity"),
        official_success=_official_success(record),
        official_reward=record.get("reward"),
        raw=record,
        steps=steps,
    )


def step_to_prompt_dict(
    step: NormalizedStep,
    *,
    include_ui: bool = True,
) -> dict[str, Any]:
    """Return the fields allowed to be sent to retrieval/judge prompts."""
    payload = {
        "step": step.step,
        "thinking": step.thinking,
        "action": step.action,
        "summary": step.summary,
        "action_target_ui": step.action_target_ui,
        "before_ui_text": _compact_ui_text(step.before_ui),
        "after_ui_text": _compact_ui_text(step.after_ui),
        "has_before_screenshot": bool(step.before_screenshot_path),
        "has_after_screenshot": bool(step.after_screenshot_path),
    }
    if not include_ui:
        payload.pop("before_ui_text")
        payload.pop("after_ui_text")
    return payload


def _extract_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    candidates = [
        trace.get("steps"),
        record.get("trace_steps"),
        record.get("steps") if isinstance(record.get("steps"), list) else None,
    ]
    for value in candidates:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_step(
    step_data: dict[str, Any],
    *,
    index: int,
    base_dir: Path | None,
    image_root: Path | None,
) -> NormalizedStep:
    action_output = str(step_data.get("action_output") or step_data.get("raw_response") or "")
    return NormalizedStep(
        step=_step_number(step_data, index),
        thinking=_first_text(
            step_data.get("thinking"),
            step_data.get("thought"),
            step_data.get("reason"),
            step_data.get("action_reason"),
            _extract_thought(action_output),
        ),
        action=_normalize_action(
            _first_value(
                step_data.get("action"),
                step_data.get("action_text"),
                step_data.get("action_output_json"),
                _extract_action(action_output),
            )
        ),
        summary=_first_text(step_data.get("summary"), step_data.get("observation_summary")),
        before_screenshot_path=_resolve_path(
            _first_text(step_data.get("before_screenshot_path")),
            base_dir=base_dir,
            image_root=image_root,
        ),
        after_screenshot_path=_resolve_path(
            _first_text(
                step_data.get("after_screenshot_path"),
                step_data.get("screenshot_path"),
                step_data.get("image_path"),
            ),
            base_dir=base_dir,
            image_root=image_root,
        ),
        before_ui=_normalize_ui_text(
            _first_value(
                step_data.get("before_ui"),
                step_data.get("before_ui_text"),
                step_data.get("ui_before"),
                step_data.get("ui_text_before"),
            )
        ),
        after_ui=_normalize_ui_text(
            _first_value(
                step_data.get("after_ui"),
                step_data.get("after_ui_text"),
                step_data.get("ui_after"),
                step_data.get("ui_text_after"),
            )
        ),
        action_target_ui=_normalize_ui_text(
            _first_value(step_data.get("action_target_ui"))
        ),
    )


def _step_number(step_data: dict[str, Any], index: int) -> int:
    value = step_data.get("step")
    try:
        return int(value)
    except (TypeError, ValueError):
        return index


def _normalize_action(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, str):
        return action.strip()
    try:
        return json.dumps(action, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(action)


def _resolve_path(
    value: str,
    *,
    base_dir: Path | None,
    image_root: Path | None,
) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return str(path)
        rebased = _rebase_missing_path(path, base_dir=base_dir, image_root=image_root)
        return str(rebased) if rebased else value
    candidates = [path]
    if image_root:
        candidates.append(image_root / path)
    if base_dir:
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else value


def _rebase_missing_path(
    path: Path,
    *,
    base_dir: Path | None,
    image_root: Path | None,
) -> Path | None:
    roots = [root for root in (image_root, base_dir) if root is not None]
    suffixes = _portable_suffixes(path, base_dir=base_dir)
    for root in roots:
        for suffix in suffixes:
            candidate = root / suffix
            if candidate.exists():
                return candidate
    return None


def _portable_suffixes(path: Path, *, base_dir: Path | None) -> list[Path]:
    parts = path.parts
    suffixes: list[Path] = []
    for marker in ("trace_images", "screenshots", "images"):
        if marker in parts:
            suffixes.append(Path(*parts[parts.index(marker) :]))
    if base_dir and base_dir.name in parts:
        index = parts.index(base_dir.name)
        suffixes.append(Path(*parts[index + 1 :]))
    return list(dict.fromkeys(suffixes))


def _official_success(record: dict[str, Any]) -> bool | None:
    if isinstance(record.get("success"), bool):
        return record["success"]
    if "is_successful" in record:
        try:
            return float(record["is_successful"]) > 0
        except (TypeError, ValueError):
            return None
    reward = record.get("reward")
    if isinstance(reward, (int, float)):
        return reward > 0
    return None


def _extract_thought(text: str) -> str:
    match = _THOUGHT_RE.search(text)
    return match.group(1).strip() if match else ""


def _extract_action(text: str) -> str:
    match = _ACTION_RE.search(text)
    return match.group(1).strip() if match else ""


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _normalize_ui_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip()).strip()
    return str(value).strip()


def _compact_ui_text(value: str, *, max_lines: int = 60, max_chars: int = 4000) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines[:max_lines])
    if len(compact) > max_chars:
        compact = compact[: max_chars - 13].rstrip() + "\n...[truncated]"
    elif len(lines) > max_lines:
        compact = compact.rstrip() + "\n...[truncated]"
    return compact
