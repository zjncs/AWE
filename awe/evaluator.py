"""Checkpoint-based LLM evaluator for AndroidWorld execution traces."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from awe.json_utils import parse_json_object
from awe.models import TextModel
from awe.prompts import (
    build_checkpoint_evaluation_prompt,
    build_checkpoint_prompt,
    build_evaluation_prompt,
)
from awe.trace_serialization import (
    build_checkpoint_payload_and_images,
    build_trial_payload_and_images,
)


CHECKPOINT_STANDARD_VERSION = 1
EVALUATION_FORMAT_VERSION = 1
CHECKPOINT_EVIDENCE_STEP_BUDGETS = (12, 8, 6, 4)
CHECKPOINT_IMAGE_BUDGETS = (6, 3, 1, 0)
# When trace has this many steps or fewer, use full-trace evaluation (v2 style)
# instead of per-checkpoint evaluation (v3 style) to avoid losing context.
FULL_TRACE_STEP_THRESHOLD = 10


class TraceEvaluator:
    """Generates stable checkpoint standards and grades traces against them."""

    def __init__(
        self,
        model: TextModel,
        *,
        checkpoint_dir: str | Path,
        regenerate_checkpoints: bool = False,
    ) -> None:
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.regenerate_checkpoints = regenerate_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Evaluates one result record."""
        task_name = str(record.get("task") or "")
        task_goal = goal_for_standard(record)

        if not (record.get("trace") or {}).get("steps"):
            return {
                "format_version": EVALUATION_FORMAT_VERSION,
                "task": task_name,
                "granularity": record.get("granularity"),
                "standard_id": build_standard_id(task_name, task_goal),
                "status": "skipped_no_trace",
                "success": None,
                "completeness_score": None,
                "checkpoint_results": [],
                "rationale": "No compact trace is attached to this trial.",
                "official_reward": record.get("reward"),
                "official_success": record.get("success"),
            }

        standard = self.load_or_create_standard(task_name, task_goal)
        trace_steps = len((record.get("trace") or {}).get("steps") or [])

        if trace_steps <= FULL_TRACE_STEP_THRESHOLD:
            evaluation = self._evaluate_full_trace(
                record=record,
                task_name=task_name,
                task_goal=task_goal,
                standard=standard,
            )
            mode = "full_trace"
        else:
            evaluation = self._evaluate_per_checkpoint(
                record=record,
                task_name=task_name,
                task_goal=task_goal,
                standard=standard,
            )
            mode = "checkpoint_evidence_v1"

        evaluation.update(
            {
                "format_version": EVALUATION_FORMAT_VERSION,
                "task": task_name,
                "granularity": record.get("granularity"),
                "standard_id": standard["standard_id"],
                "goal_hash": standard["goal_hash"],
                "status": "evaluated",
                "evaluation_mode": mode,
                "trace_steps": trace_steps,
                "official_reward": record.get("reward"),
                "official_success": record.get("success"),
                "agreement_with_reward": _agreement_with_reward(
                    evaluation.get("success"), record.get("reward")
                ),
            }
        )
        return evaluation

    def _evaluate_full_trace(
        self,
        *,
        record: dict[str, Any],
        task_name: str,
        task_goal: str,
        standard: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluates all checkpoints in one LLM call with the full trace."""
        trial_payload, prompt_images = build_trial_payload_and_images(record)
        prompt = build_evaluation_prompt(
            task_name=task_name,
            task_goal=task_goal,
            checkpoint_standard=standard,
            trial_payload=trial_payload,
        )
        raw = self.model.complete(prompt, images=prompt_images)
        parsed = parse_json_object(raw)
        evaluation = _normalize_evaluation(parsed, standard)
        # Ensure checkpoint_results have the extra fields for consistency
        for cp_result in evaluation.get("checkpoint_results", []):
            cp_result.setdefault("evidence_steps", list(range(1, len((record.get("trace") or {}).get("steps") or []) + 1)))
            cp_result.setdefault("evidence_images", len(prompt_images))
            cp_result.setdefault("insufficient_trace", False)
        return evaluation

    def _evaluate_per_checkpoint(
        self,
        *,
        record: dict[str, Any],
        task_name: str,
        task_goal: str,
        standard: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluates each checkpoint individually with relevance-filtered evidence."""
        checkpoint_results = []
        insufficient_trace = False
        for checkpoint in standard.get("checkpoints", []):
            checkpoint_result = self._evaluate_checkpoint(
                record=record,
                task_name=task_name,
                task_goal=task_goal,
                checkpoint=checkpoint,
                checkpoint_standard=standard,
            )
            checkpoint_results.append(checkpoint_result)
            insufficient_trace = insufficient_trace or checkpoint_result["insufficient_trace"]
        return _aggregate_checkpoint_results(
            checkpoint_results,
            standard,
            insufficient_trace=insufficient_trace,
        )

    def _evaluate_checkpoint(
        self,
        *,
        record: dict[str, Any],
        task_name: str,
        task_goal: str,
        checkpoint: dict[str, Any],
        checkpoint_standard: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        attempted: set[tuple[int, int]] = set()
        attempted_payloads: set[tuple[tuple[Any, ...], tuple[str, ...]]] = set()
        for max_steps in CHECKPOINT_EVIDENCE_STEP_BUDGETS:
            for max_images in CHECKPOINT_IMAGE_BUDGETS:
                budget = (max_steps, max_images)
                if budget in attempted:
                    continue
                attempted.add(budget)
                evidence_payload, prompt_images = build_checkpoint_payload_and_images(
                    record,
                    checkpoint,
                    max_steps=max_steps,
                    max_images=max_images,
                )
                payload_signature = (
                    tuple(evidence_payload.get("selected_step_numbers") or []),
                    tuple(prompt_images),
                )
                if payload_signature in attempted_payloads:
                    continue
                attempted_payloads.add(payload_signature)
                prompt = build_checkpoint_evaluation_prompt(
                    task_name=task_name,
                    task_goal=task_goal,
                    checkpoint=checkpoint,
                    checkpoint_standard=checkpoint_standard,
                    evidence_payload=evidence_payload,
                )
                try:
                    raw = self.model.complete(prompt, images=prompt_images)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    last_error = exc
                    if _is_prompt_too_large_error(exc):
                        continue
                    raise
                parsed = parse_json_object(raw)
                return _normalize_checkpoint_result(
                    parsed,
                    checkpoint,
                    evidence_payload=evidence_payload,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Checkpoint evaluation exhausted all budgets without a result.")

    def evaluate_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Evaluates records in order."""
        evaluations = []
        for record in records:
            try:
                evaluations.append(self.evaluate_record(record))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                evaluations.append(_evaluation_error_result(record, exc))
        return evaluations

    def load_or_create_standard(self, task_name: str, task_goal: str) -> dict[str, Any]:
        """Loads a cached checkpoint standard or asks the model to create one."""
        standard_id = build_standard_id(task_name, task_goal)
        path = self.checkpoint_dir / f"{standard_id}.json"
        if path.exists() and not self.regenerate_checkpoints:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                pass

        raw = self.model.complete(build_checkpoint_prompt(task_name, task_goal))
        parsed = parse_json_object(raw)
        standard = _normalize_checkpoint_standard(parsed, task_name, task_goal, standard_id)
        _atomic_write_text(
            path,
            json.dumps(standard, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return standard


def goal_for_standard(record: dict[str, Any]) -> str:
    """Returns the task goal used for shared standards.

    ``base_goal`` intentionally excludes workflow/action guidance, so different
    prompt granularities for the same sampled task share the same rubric.
    """
    return str(record.get("base_goal") or record.get("goal") or record.get("goal_used") or "")


def build_standard_id(task_name: str, task_goal: str) -> str:
    payload = json.dumps(
        {
            "version": CHECKPOINT_STANDARD_VERSION,
            "task_name": task_name,
            "task_goal": task_goal,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _normalize_checkpoint_standard(
    parsed: dict[str, Any],
    task_name: str,
    task_goal: str,
    standard_id: str,
) -> dict[str, Any]:
    checkpoints = parsed.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Checkpoint generation returned no checkpoints.")

    normalized = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        if not isinstance(checkpoint, dict):
            continue
        normalized.append(
            {
                "id": str(checkpoint.get("id") or f"cp{index}"),
                "description": str(checkpoint.get("description") or "").strip(),
                "required": bool(checkpoint.get("required", True)),
                "evidence_hint": str(checkpoint.get("evidence_hint") or "").strip(),
            }
        )
    if not normalized:
        raise ValueError("Checkpoint generation returned malformed checkpoints.")

    return {
        "format_version": CHECKPOINT_STANDARD_VERSION,
        "standard_id": standard_id,
        "goal_hash": hashlib.sha256(task_goal.encode("utf-8")).hexdigest()[:16],
        "task_name": task_name,
        "task_goal": task_goal,
        "task_goal_rewrite": str(parsed.get("task_goal_rewrite") or "").strip(),
        "checkpoints": normalized,
        "success_rule": str(parsed.get("success_rule") or "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_evaluation(
    parsed: dict[str, Any],
    standard: dict[str, Any],
) -> dict[str, Any]:
    by_id = {
        str(item.get("id")): item
        for item in parsed.get("checkpoint_results", [])
        if isinstance(item, dict)
    }
    checkpoint_results = []
    achieved_required = 0
    total_required = 0

    for checkpoint in standard.get("checkpoints", []):
        checkpoint_id = str(checkpoint.get("id"))
        result = by_id.get(checkpoint_id, {})
        achieved = bool(result.get("achieved", False))
        required = bool(checkpoint.get("required", True))
        if required:
            total_required += 1
            achieved_required += int(achieved)
        checkpoint_results.append(
            {
                "id": checkpoint_id,
                "achieved": achieved,
                "confidence": _clamp_float(result.get("confidence"), 0.0, 1.0),
                "evidence": str(result.get("evidence") or "").strip(),
                "missing_or_conflict": str(result.get("missing_or_conflict") or "").strip(),
            }
        )

    computed_score = achieved_required / total_required if total_required else 0.0
    success = (total_required > 0) and (achieved_required == total_required)

    return {
        "checkpoint_results": checkpoint_results,
        "success": success,
        "completeness_score": _clamp_float(
            parsed.get("completeness_score"), 0.0, 1.0, default=computed_score
        ),
        "insufficient_trace": bool(parsed.get("insufficient_trace", False)),
        "rationale": str(parsed.get("rationale") or "").strip(),
    }


def _normalize_checkpoint_result(
    parsed: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": str(parsed.get("id") or checkpoint.get("id") or "").strip(),
        "achieved": bool(parsed.get("achieved", False)),
        "confidence": _clamp_float(parsed.get("confidence"), 0.0, 1.0),
        "evidence": str(parsed.get("evidence") or "").strip(),
        "missing_or_conflict": str(parsed.get("missing_or_conflict") or "").strip(),
        "insufficient_trace": bool(parsed.get("insufficient_trace", False)),
        "evidence_steps": list(evidence_payload.get("selected_step_numbers") or []),
        "evidence_images": int(evidence_payload.get("trace_screenshots_attached") or 0),
    }


def _is_prompt_too_large_error(error: Exception) -> bool:
    text = str(error).lower()
    return "max message tokens" in text or "total tokens of image and text exceed" in text


def _aggregate_checkpoint_results(
    checkpoint_results: list[dict[str, Any]],
    standard: dict[str, Any],
    *,
    insufficient_trace: bool,
) -> dict[str, Any]:
    required_ids = {
        str(checkpoint.get("id"))
        for checkpoint in standard.get("checkpoints", [])
        if bool(checkpoint.get("required", True))
    }
    achieved_required = sum(
        1
        for result in checkpoint_results
        if result["id"] in required_ids and result.get("achieved")
    )
    total_required = len(required_ids)
    success = (total_required > 0) and (achieved_required == total_required)
    completeness_score = achieved_required / total_required if total_required else 0.0
    missing_required = [
        result["id"]
        for result in checkpoint_results
        if result["id"] in required_ids and not result.get("achieved")
    ]

    if success:
        rationale = "All required checkpoints were supported by the selected evidence."
    elif insufficient_trace and missing_required:
        rationale = (
            "Missing required checkpoints under limited evidence: "
            + ", ".join(missing_required)
        )
    elif missing_required:
        rationale = "Missing required checkpoints: " + ", ".join(missing_required)
    else:
        rationale = "No required checkpoints were available."

    return {
        "checkpoint_results": checkpoint_results,
        "success": success,
        "completeness_score": completeness_score,
        "insufficient_trace": insufficient_trace,
        "rationale": rationale,
    }


def _clamp_float(
    value: Any,
    low: float,
    high: float,
    *,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, low), high)


def _agreement_with_reward(eval_success: Any, reward: Any) -> bool | None:
    if eval_success is None or reward is None:
        return None
    try:
        reward_success = float(reward) >= 0.5
    except (TypeError, ValueError):
        return None
    return bool(eval_success) == reward_success


def _evaluation_error_result(record: dict[str, Any], error: Exception) -> dict[str, Any]:
    task_name = str(record.get("task") or "")
    task_goal = goal_for_standard(record)
    return {
        "format_version": EVALUATION_FORMAT_VERSION,
        "task": task_name,
        "granularity": record.get("granularity"),
        "standard_id": build_standard_id(task_name, task_goal),
        "status": "evaluation_error",
        "success": None,
        "completeness_score": None,
        "checkpoint_results": [],
        "rationale": f"{type(error).__name__}: {error}",
        "official_reward": record.get("reward"),
        "official_success": record.get("success"),
        "agreement_with_reward": None,
    }


def _atomic_write_text(path: Path, content: str, *, encoding: str) -> None:
    """Atomically writes text to ``path`` in a temp file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
