"""Prompt builders for checkpoint generation and checkpoint-focused grading."""

from __future__ import annotations

import json
from typing import Any


def build_checkpoint_prompt(task_name: str, task_goal: str) -> str:
    """Builds the prompt that turns a task goal into stable checkpoints."""
    return f"""CHECKPOINT_GENERATION
You are defining a stable grading rubric for AndroidWorld task executions.

Rules:
- Use only the task name and task goal. Do not assume facts from any execution trace.
- The same task goal must always produce the same grading standard.
- Make 3 to 7 checkpoints.
- Each checkpoint must be a single observable requirement needed for task success.
- Include concrete values from the task goal when present: names, numbers, dates, text, files, folders, durations, message bodies.
- Do not add extra requirements that are merely one possible UI path unless the goal requires them.
- Mark checkpoints as required unless they are truly optional.
- Do not split a single field (e.g. full name) into sub-fields unless the goal explicitly requires it. If the goal says "Alex Chen", treat the full name as one requirement, not separate first-name and last-name requirements.

RESPOND WITH A JSON OBJECT ONLY. NO MARKDOWN FENCES. NO EXPLANATORY TEXT BEFORE OR AFTER THE JSON.

Task name: {task_name}
Task goal: {task_goal}

Required JSON schema:
{{
  "task_name": "{task_name}",
  "task_goal_rewrite": "one sentence restatement of the target final state",
  "checkpoints": [
    {{
      "id": "cp1",
      "description": "observable requirement",
      "required": true,
      "evidence_hint": "what evidence in an execution trace would satisfy it"
    }}
  ],
  "success_rule": "how to decide final success from the checkpoints"
}}
"""


def build_evaluation_prompt(
    *,
    task_name: str,
    task_goal: str,
    checkpoint_standard: dict[str, Any],
    trial_payload: dict[str, Any],
) -> str:
    """Builds the prompt that grades one execution trace with a fixed rubric."""
    standard_json = json.dumps(checkpoint_standard, ensure_ascii=False, indent=2)
    trial_json = json.dumps(trial_payload, ensure_ascii=False, indent=2)
    return f"""TRACE_EVALUATION
You are a strict AndroidWorld trace evaluator. Grade this single execution using
the checkpoint standard below. The standard is fixed and must not be changed.

Rules:
- Judge from the trace evidence, not from the agent saying it is done.
- A checkpoint is achieved only when the trace gives positive evidence.
- If the trace is too thin, mark the relevant checkpoint false with low confidence.
- Do not reward irrelevant progress.
- Use the official reward only as auxiliary context; do not let it override trace evidence.
- Final success must be true only if all required checkpoints are achieved and there is no conflicting evidence.
- The same user message may also include screenshots referenced by
  `trace_screenshot_manifest`. Use them as auxiliary evidence when useful.

RESPOND WITH A JSON OBJECT ONLY. NO MARKDOWN FENCES. NO EXPLANATORY TEXT BEFORE OR AFTER THE JSON.

Task name: {task_name}
Task goal: {task_goal}

Fixed checkpoint standard:
{standard_json}

Execution trace:
{trial_json}

Required JSON schema:
{{
  "checkpoint_results": [
    {{
      "id": "cp1",
      "achieved": true,
      "confidence": 0.0,
      "evidence": "short trace-grounded evidence",
      "missing_or_conflict": "what is missing or conflicting, empty if none"
    }}
  ],
  "success": false,
  "completeness_score": 0.0,
  "insufficient_trace": false,
  "rationale": "brief final judgment"
}}
"""


def build_checkpoint_evaluation_prompt(
    *,
    task_name: str,
    task_goal: str,
    checkpoint: dict[str, Any],
    checkpoint_standard: dict[str, Any],
    evidence_payload: dict[str, Any],
) -> str:
    """Builds the prompt that grades one checkpoint against selected evidence."""
    checkpoint_json = json.dumps(checkpoint, ensure_ascii=False, indent=2)
    standard_json = json.dumps(
        {
            "standard_id": checkpoint_standard.get("standard_id"),
            "success_rule": checkpoint_standard.get("success_rule"),
            "task_goal_rewrite": checkpoint_standard.get("task_goal_rewrite"),
        },
        ensure_ascii=False,
        indent=2,
    )
    evidence_json = json.dumps(evidence_payload, ensure_ascii=False, indent=2)
    return f"""CHECKPOINT_EVALUATION
You are grading one checkpoint for an AndroidWorld execution. Evaluate ONLY the
checkpoint under review using the evidence subset below.

Rules:
- Judge from the provided evidence subset, not from the task goal alone.
- The evidence subset was preselected for relevance; do not assume omitted steps
  were successful.
- Mark achieved=true only when the evidence gives positive support.
- If the evidence is weak or ambiguous, set achieved=false and
  insufficient_trace=true with low confidence.
- The same user message may also include screenshots referenced by
  `trace_screenshot_manifest`. Use them as auxiliary evidence when useful.
- Do not grade any checkpoint other than the one under review.

RESPOND WITH A JSON OBJECT ONLY. NO MARKDOWN FENCES. NO EXPLANATORY TEXT BEFORE OR AFTER THE JSON.

Task name: {task_name}
Task goal: {task_goal}

Shared evaluation context:
{standard_json}

Checkpoint under review:
{checkpoint_json}

Evidence subset:
{evidence_json}

Required JSON schema:
{{
  "id": "{checkpoint.get("id", "cp1")}",
  "achieved": false,
  "confidence": 0.0,
  "evidence": "short evidence grounded in the subset",
  "missing_or_conflict": "what is missing or conflicting, empty if none",
  "insufficient_trace": false
}}
"""
