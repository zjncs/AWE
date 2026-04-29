"""Prompt text for the standalone evaluator."""

from __future__ import annotations

import json
from typing import Any

from gui_trace_evaluator.record_adapter import NormalizedRecord, NormalizedStep, step_to_prompt_dict


ANDROID_WORLD_CONTEXT = """AndroidWorld/AWE environment context:
- The Android shared external storage is canonically accessible as /sdcard.
- A task goal may mention an emulator storage label such as sdk_gphone_x86_64, while the device UI may display a different ABI-specific label such as sdk_gphone64_arm64.
- Treat these labels as device/storage presentation details when the trace and read-only evidence refer to the same shared Android storage or public folder.
- Do not ignore exact target values such as filenames, contact names, dates, note text, or app-specific content.
"""


def checkpoint_generation_request(record: NormalizedRecord) -> list[dict[str, Any]]:
    """Build checkpoint generation messages."""
    content = f"""CHECKPOINT_GENERATION
Create a stable grading rubric for one GUI task execution.

Rules:
- Use only the task and goal. Do not inspect any trace.
- Make 3 to 5 observable checkpoints for most tasks.
- Use exactly 2 checkpoints only for truly atomic one-toggle/one-tap tasks
  where a single final state almost fully determines correctness (for example:
  turning Wi-Fi off/on with no additional constraints).
- Include concrete values from the goal when present.
- Mark checkpoints required unless genuinely optional.
- The same task goal should produce the same standard across runs.
- Prefer outcome/final-state checkpoints over low-level process checkpoints.
- Keep checkpoints generic and reusable across apps with similar goals.
- Avoid brittle app-specific UI mechanics or route details.
- Do not make pure setup/navigation steps required unless explicitly asked in the goal.
- Focus on final outcome first; add intermediate checkpoints only when needed for reliable verification.
- Each checkpoint must include a checkpoint_type from: outcome, consistency, supporting.
- Assign a relative weight (0.1 to 1.0) for each checkpoint importance.


Respond with JSON only.

Task: {record.task}
Goal: {record.goal}
{ANDROID_WORLD_CONTEXT}

Required schema:
{{
  "task_goal_rewrite": "one sentence target final state",
  "checkpoints": [
    {{
      "id": "cp1",
      "description": "observable requirement",
      "required": true,
      "evidence_hint": "what trace/screenshot evidence would satisfy it",
      "checkpoint_type": "outcome",
      "weight": 1.0
    }}
  ],
  "success_rule": "how final success is decided"
}}
"""
    return [{"role": "user", "content": content}]


def retrieval_instruction(record: NormalizedRecord, checkpoint: dict[str, Any]) -> str:
    """Initial official-style instruction for step retrieval."""
    return (
        "Evaluate a completed phone GUI agent trace. Select evidence steps for "
        f"this checkpoint. Task: {record.task}. Goal: {record.goal}. "
        f"{ANDROID_WORLD_CONTEXT} "
        f"Checkpoint: {json.dumps(checkpoint, ensure_ascii=False)}"
    )


def retrieval_final_request(
    *,
    record: NormalizedRecord,
    checkpoint: dict[str, Any],
    steps: list[NormalizedStep],
    max_selected_steps: int,
    repair_context: dict[str, Any] | None = None,
) -> str:
    """Final JSON request for model-based step retrieval."""
    step_payload = [step_to_prompt_dict(step, include_ui=False) for step in steps]
    repair_text = ""
    if repair_context:
        repair_text = (
            "\nPrevious retrieval was not trusted. Do a second-pass repair retrieval. "
            "Broaden the step set if needed, include final verification/status steps "
            "when useful, and explain whether the repaired selection is trustworthy.\n"
            f"Previous retrieval: {json.dumps(repair_context, ensure_ascii=False)}\n"
        )
    return f"""STEP_RETRIEVAL
Use the lightweight step summaries below to select evidence steps. Screenshots and full UI tables are intentionally not included in this stage.
{repair_text}
Select at most {max_selected_steps} step numbers that are useful evidence for the checkpoint.
Set trusted=false if the selected steps still cannot support a reliable screenshot-based judgment.
Do not return an empty selected_steps list unless the trace has zero steps.
Prefer steps that contain the decisive action, the immediate before/after state, and the final visible state relevant to the checkpoint.
If the checkpoint is about a final outcome, include the last relevant state-changing action and any later step that can show the outcome.
If there is no conclusive success evidence, still select the best negative-evidence steps:
failed attempts, repeated attempts, relevant navigation attempts, error states, or final visible states.
The judge model will decide whether these steps prove success, prove failure, or remain insufficient.

Task: {record.task}
Goal: {record.goal}
{ANDROID_WORLD_CONTEXT}
Checkpoint: {json.dumps(checkpoint, ensure_ascii=False, indent=2)}
Available steps:
{json.dumps(step_payload, ensure_ascii=False, indent=2)}

Respond with JSON only:
{{
  "selected_steps": [1, 2],
  "trusted": true,
  "confidence": 0.0,
  "rationale": "why these steps are relevant",
  "fallback_reason": "why retrieval remains unreliable, empty if trusted"
}}
"""


def checkpoint_judge_instruction(record: NormalizedRecord, checkpoint: dict[str, Any]) -> str:
    """Initial official-style instruction for screenshot-based checkpoint judging."""
    return (
        "Evaluate a completed phone GUI agent trace using selected evidence screenshots. "
        f"Task: {record.task}. Goal: {record.goal}. "
        f"{ANDROID_WORLD_CONTEXT} "
        f"Checkpoint: {json.dumps(checkpoint, ensure_ascii=False)}"
    )


def checkpoint_judge_final_request(
    *,
    record: NormalizedRecord,
    checkpoint: dict[str, Any],
    selected_steps: list[NormalizedStep],
    image_manifest: list[dict[str, Any]],
    retrieval: dict[str, Any],
    read_tool_results: list[dict[str, Any]] | None = None,
    first_pass_result: dict[str, Any] | None = None,
) -> str:
    """Final JSON request for one checkpoint judgment."""
    tool_text = ""
    if read_tool_results is None:
        tool_text = """
If screenshots and action history are not enough to determine the real final state, request read-only fallback tools instead of guessing.
Available read-only tools:
- list_dir: {"tool": "list_dir", "path": "task-relevant Android path", "reason": "..."}
- stat_path: {"tool": "stat_path", "path": "task-relevant Android path", "reason": "..."}
- find_file: {"tool": "find_file", "root": "task-relevant Android root", "name": "exact filename", "reason": "..."}
- read_text_file: {"tool": "read_text_file", "path": "task-relevant Android path", "reason": "..."}
- query_app_sqlite: {"tool": "query_app_sqlite", "package": "allowed app package", "db_path": "/data/data/<package>/...", "query": "SELECT ...", "limit": 100, "reason": "..."}
Only request tools when they are necessary for final-state verification. Tool requests are evidence requests, not the final answer.
"""
    else:
        compact_read_results = _compact_read_tool_results(read_tool_results)
        tool_text = f"""
READ_TOOL_RESULTS
The following read-only tool results were collected after execution. They are additional final-state evidence, not rule-based labels.
Combine them with screenshots and action history to make the final checkpoint judgment. Do not request more tools in this pass.
First-pass judgment before tool results: {json.dumps(first_pass_result or {}, ensure_ascii=False, indent=2)}
Read-only tool results: {json.dumps(compact_read_results, ensure_ascii=False, indent=2)}
"""
    return f"""CHECKPOINT_EVALUATION
Judge only the checkpoint below from the selected Thought/Action history and attached screenshots.
Do not assume omitted steps succeeded. If evidence is weak, mark achieved=false and insufficient_trace=true.
The retrieval metadata is only a candidate-selection signal. You must make the final judgment yourself.
If retrieval_trusted=false but the selected steps/screenshots are still enough, you may mark achieved=true.
If selected steps are empty or screenshots are missing, decide from the available evidence and explain the gap.
Treat Thought/Action/Observation Summary as agent claims, not ground truth. When a claim conflicts with a screenshot, the screenshot wins.
For visual or text-placement requirements, inspect the screenshot directly: order, position, selected file/app, final visible state, and any error/disabled state matter.
If the goal requires a value to be at the top/first/last/specific location, verify that exact placement visually. If the screenshot shows a different placement, mark achieved=false.
If UI text is truncated, hidden, or duplicated by prefix, do not infer the full identity from the prefix alone. Use non-conflicting action history and before/after visual changes to disambiguate; if still ambiguous, explain the ambiguity instead of claiming the full target is present.
If a goal outcome is already unambiguous from the final state, do not fail a checkpoint only because an intermediate "proof step" is missing (for example, contact details page proving creation even if list page is not shown; target file absent at exact path even if full filename was truncated earlier).
For file/save tasks, a clicked save action is not enough by itself. Require visible final-state evidence or clearly explain why persistence is still uncertain.
For persistent-state tasks, stale or unrefreshed UI can conflict with the real final state. If final state cannot be determined from screenshots, set needs_fallback_verification=true and request read-only tools.
Return both achieved and score:
- score in [0, 1], where 1.0 means fully satisfied, 0.5 means partially satisfied with meaningful progress, 0.0 means not satisfied.
- achieved=true only when score >= 0.8 with clear evidence.
{tool_text}

Task: {record.task}
Goal: {record.goal}
{ANDROID_WORLD_CONTEXT}
Checkpoint: {json.dumps(checkpoint, ensure_ascii=False, indent=2)}
Selected steps: {[step.step for step in selected_steps]}
Screenshot manifest: {json.dumps(image_manifest, ensure_ascii=False, indent=2)}
Retrieval metadata: {json.dumps(retrieval, ensure_ascii=False, indent=2)}

Respond with JSON only:
{{
  "id": "{checkpoint.get("id", "cp1")}",
  "achieved": false,
  "score": 0.0,
  "confidence": 0.0,
  "evidence": "short evidence grounded in selected steps/screenshots",
  "missing_or_conflict": "missing or conflicting evidence, empty if none",
  "insufficient_trace": false,
  "needs_fallback_verification": false,
  "read_requests": []
}}
"""


def _compact_read_tool_results(read_tool_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Summarize read-tool evidence to reduce prompt token usage."""
    if not read_tool_results:
        return []
    compacted: list[dict[str, Any]] = []
    for item in read_tool_results:
        if not isinstance(item, dict):
            continue
        compact: dict[str, Any] = {
            "type": item.get("type"),
            "tool": item.get("tool"),
            "status": item.get("status"),
            "request": item.get("request", {}),
            "source": item.get("source"),
        }
        for key in (
            "exists",
            "found",
            "matches",
            "interpretation",
            "returncode",
            "row_count",
            "screenshot_path",
        ):
            if key in item:
                compact[key] = item.get(key)
        output = str(item.get("output") or "").strip()
        if output:
            compact["output_preview"] = _trim_text(output, max_chars=280)
        final_ui = str(item.get("final_ui") or "").strip()
        if final_ui:
            compact["final_ui_preview"] = _trim_text(final_ui, max_chars=280)
        compacted.append(compact)
    return compacted


def _trim_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "...[truncated]"
