"""Checkpoint evaluator using official-style Doubao GUI messages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gui_trace_evaluator.json_utils import parse_json_object
from gui_trace_evaluator.models import ChatModel
from gui_trace_evaluator.official_messages import build_trace_messages
from gui_trace_evaluator.prompts import (
    checkpoint_generation_request,
    checkpoint_judge_final_request,
    checkpoint_judge_instruction,
    retrieval_final_request,
    retrieval_instruction,
)
from gui_trace_evaluator.read_tools import (
    ReadToolConfig,
    ReadToolRunner,
    default_read_requests,
)
from gui_trace_evaluator.record_adapter import (
    NormalizedRecord,
    NormalizedStep,
    normalize_record,
)


CHECKPOINT_STANDARD_VERSION = 1
EVALUATION_FORMAT_VERSION = 1


class TraceEvaluator:
    """Standalone GUI trace evaluator."""

    def __init__(
        self,
        model: ChatModel,
        *,
        checkpoint_dir: str | Path,
        image_root: str | Path | None = None,
        regenerate_checkpoints: bool = False,
        max_selected_steps: int = 30,
        max_screenshot_turns: int = 10,
        max_retrieval_trace_steps: int = 40,
        retrieval_min_confidence: float = 0.55,
        fallback_confidence_threshold: float = 0.7,
        read_tool_runner: ReadToolRunner | None = None,
        read_tool_config: ReadToolConfig | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.image_root = Path(image_root).resolve() if image_root else None
        self.regenerate_checkpoints = regenerate_checkpoints
        self.max_selected_steps = max_selected_steps
        self.max_screenshot_turns = max_screenshot_turns
        self.max_retrieval_trace_steps = max_retrieval_trace_steps
        self.retrieval_min_confidence = retrieval_min_confidence
        self.fallback_confidence_threshold = fallback_confidence_threshold
        self.read_tool_runner = read_tool_runner or ReadToolRunner(read_tool_config)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_records(
        self,
        records: list[dict[str, Any]],
        *,
        base_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        return [self.evaluate_record(record, base_dir=base_dir) for record in records]

    def evaluate_record(
        self,
        record_data: dict[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        record = normalize_record(record_data, base_dir=base_dir, image_root=self.image_root)
        standard_id = build_standard_id(record.task, record.goal)
        if not record.steps:
            return {
                "format_version": EVALUATION_FORMAT_VERSION,
                "task": record.task,
                "granularity": record.granularity,
                "standard_id": standard_id,
                "status": "skipped_no_trace",
                "success": None,
                "completeness_score": None,
                "checkpoint_results": [],
                "rationale": "No trace steps were found.",
                "official_reward": record.official_reward,
                "official_success": record.official_success,
            }

        try:
            standard = self.load_or_create_standard(record)
            return self._evaluate_with_standard(record, standard)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if _is_token_limit_error(exc):
                _progress(
                    f"[evaluator] {record.task}: token limit hit, retrying with compact context"
                )
                original_steps = self.max_selected_steps
                original_turns = self.max_screenshot_turns
                original_retrieval_steps = self.max_retrieval_trace_steps
                self.max_selected_steps = min(self.max_selected_steps, 8)
                self.max_screenshot_turns = min(self.max_screenshot_turns, 3)
                self.max_retrieval_trace_steps = min(self.max_retrieval_trace_steps, 20)
                try:
                    standard = self.load_or_create_standard(record)
                    retried = self._evaluate_with_standard(record, standard)
                    retried["evaluation_mode"] = "doubao_official_gui_messages_v1_compact_retry"
                    retried["compact_retry_used"] = True
                    retried["compact_retry_config"] = {
                        "max_selected_steps": self.max_selected_steps,
                        "max_screenshot_turns": self.max_screenshot_turns,
                    }
                    return retried
                except Exception as retry_exc:  # pylint: disable=broad-exception-caught
                    exc = retry_exc
                finally:
                    self.max_selected_steps = original_steps
                    self.max_screenshot_turns = original_turns
                    self.max_retrieval_trace_steps = original_retrieval_steps
            return {
                "format_version": EVALUATION_FORMAT_VERSION,
                "task": record.task,
                "granularity": record.granularity,
                "standard_id": standard_id,
                "status": "evaluation_error",
                "success": None,
                "completeness_score": None,
                "checkpoint_results": [],
                "rationale": str(exc),
                "official_reward": record.official_reward,
                "official_success": record.official_success,
                "agreement_with_reward": None,
            }

    def _evaluate_with_standard(
        self,
        record: NormalizedRecord,
        standard: dict[str, Any],
    ) -> dict[str, Any]:
        checkpoints = standard.get("checkpoints", [])
        _progress(f"[evaluator] {record.task}: evaluating {len(checkpoints)} checkpoints")
        checkpoint_results = []
        for index, checkpoint in enumerate(checkpoints, start=1):
            _progress(
                f"[evaluator] checkpoint {index}/{len(checkpoints)} "
                f"{checkpoint.get('id', '')}: {checkpoint.get('description', '')[:120]}"
            )
            checkpoint_results.append(self._evaluate_checkpoint(record, checkpoint))
        evaluation = _aggregate_checkpoint_results(checkpoint_results, standard)
        evaluation.update(
            {
                "format_version": EVALUATION_FORMAT_VERSION,
                "task": record.task,
                "granularity": record.granularity,
                "standard_id": standard["standard_id"],
                "goal_hash": standard["goal_hash"],
                "status": "evaluated",
                "evaluation_mode": "doubao_official_gui_messages_v1",
                "trace_steps": len(record.steps),
                "official_reward": record.official_reward,
                "official_success": record.official_success,
                "agreement_with_reward": _agreement_with_reward(
                    evaluation.get("success"),
                    record.official_success,
                ),
                "agreement_with_reward_band": _agreement_with_reward_band(
                    evaluation.get("predicted_reward"),
                    record.official_reward,
                ),
            }
        )
        return evaluation

    def load_or_create_standard(self, record: NormalizedRecord) -> dict[str, Any]:
        standard_id = build_standard_id(record.task, record.goal)
        path = self.checkpoint_dir / f"{standard_id}.json"
        if path.exists() and not self.regenerate_checkpoints:
            return json.loads(path.read_text(encoding="utf-8"))

        raw = self.model.complete(checkpoint_generation_request(record))
        parsed = parse_json_object(raw)
        standard = _normalize_standard(parsed, task=record.task, goal=record.goal)
        _atomic_write_text(path, json.dumps(standard, ensure_ascii=False, indent=2))
        return standard

    def _evaluate_checkpoint(
        self,
        record: NormalizedRecord,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        retrieval = self._retrieve_steps(record, checkpoint)
        if not _trusted_retrieval(retrieval, self.retrieval_min_confidence):
            repair = self._retrieve_steps(record, checkpoint, repair_context=retrieval)
            retrieval = _choose_retrieval_for_judge(retrieval, repair)

        selected_steps = _steps_by_number(
            record.steps,
            _with_neighbor_context(retrieval.get("selected_steps") or [], record.steps),
        )
        result, manifest, parsed = self._judge_checkpoint(
            record=record,
            checkpoint=checkpoint,
            selected_steps=selected_steps,
            retrieval=retrieval,
        )
        read_requests = _normalize_read_requests(parsed)
        if _should_run_read_tools(
            result,
            record=record,
            checkpoint=checkpoint,
            fallback_confidence_threshold=self.fallback_confidence_threshold,
        ):
            first_pass_result = dict(result)
            if not read_requests:
                read_requests = default_read_requests(record, checkpoint)
            stored_results = _stored_read_tool_results(record)
            tool_results = (
                stored_results
                if stored_results
                else self.read_tool_runner.run_requests(record, checkpoint, read_requests)
            )
            second_result, manifest, second_parsed = self._judge_checkpoint(
                record=record,
                checkpoint=checkpoint,
                selected_steps=selected_steps,
                retrieval=retrieval,
                read_tool_results=tool_results,
                first_pass_result=first_pass_result,
            )
            result = second_result
            parsed = second_parsed
            result["read_tool_verification"] = {
                "triggered": True,
                "requests": read_requests,
                "results": tool_results,
                "source": "record_post_execution_evidence" if stored_results else "live_read_tools",
                "first_pass_result": _compact_checkpoint_result_for_metadata(first_pass_result),
            }
        else:
            result["read_tool_verification"] = {
                "triggered": False,
                "requests": read_requests,
                "results": [],
            }

        result.update(
            _checkpoint_metadata(
                selected_steps=selected_steps,
                manifest=manifest,
                retrieval=retrieval,
                parsed=parsed,
            )
        )
        return result

    def _judge_checkpoint(
        self,
        *,
        record: NormalizedRecord,
        checkpoint: dict[str, Any],
        selected_steps: list[NormalizedStep],
        retrieval: dict[str, Any],
        read_tool_results: list[dict[str, Any]] | None = None,
        first_pass_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        image_numbers = {step.step for step in selected_steps if step.evidence_screenshot_path}
        messages, manifest = build_trace_messages(
            instruction=checkpoint_judge_instruction(record, checkpoint),
            steps=selected_steps,
            final_request=checkpoint_judge_final_request(
                record=record,
                checkpoint=checkpoint,
                selected_steps=selected_steps,
                image_manifest=manifest_placeholder(selected_steps),
                retrieval=retrieval,
                read_tool_results=read_tool_results,
                first_pass_result=first_pass_result,
            ),
            image_step_numbers=image_numbers,
            max_screenshot_turns=self.max_screenshot_turns,
        )
        messages[-1]["content"] = checkpoint_judge_final_request(
            record=record,
            checkpoint=checkpoint,
            selected_steps=selected_steps,
            image_manifest=manifest,
            retrieval=retrieval,
            read_tool_results=read_tool_results,
            first_pass_result=first_pass_result,
        )
        raw = self.model.complete(messages)
        parsed = parse_json_object(raw)
        return _normalize_checkpoint_result(parsed, checkpoint), manifest, parsed

    def _retrieve_steps(
        self,
        record: NormalizedRecord,
        checkpoint: dict[str, Any],
        *,
        repair_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieval_steps = _sample_steps_for_retrieval(
            record.steps,
            limit=self.max_retrieval_trace_steps,
        )
        messages, _ = build_trace_messages(
            instruction=retrieval_instruction(record, checkpoint),
            steps=retrieval_steps,
            final_request=retrieval_final_request(
                record=record,
                checkpoint=checkpoint,
                steps=retrieval_steps,
                max_selected_steps=self.max_selected_steps,
                repair_context=repair_context,
            ),
            image_step_numbers=set(),
            max_screenshot_turns=0,
        )
        raw = self.model.complete(messages)
        parsed = parse_json_object(raw)
        retrieval = _normalize_retrieval(parsed)
        retrieval["method"] = (
            "model_step_retrieval_repair_v1"
            if repair_context
            else "model_step_retrieval_v1"
        )
        return retrieval


def build_standard_id(task_name: str, task_goal: str) -> str:
    digest = hashlib.sha256(f"{task_name}\n{task_goal}".encode("utf-8")).hexdigest()
    return digest[:20]


def manifest_placeholder(steps: list[NormalizedStep]) -> list[dict[str, Any]]:
    return [
        {
            "step": step.step,
            "kind": step.evidence_screenshot_kind,
            "has_image": bool(step.evidence_screenshot_path),
        }
        for step in steps
    ]


def _normalize_standard(parsed: dict[str, Any], *, task: str, goal: str) -> dict[str, Any]:
    checkpoints = parsed.get("checkpoints") if isinstance(parsed.get("checkpoints"), list) else []
    normalized = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        if not isinstance(checkpoint, dict):
            continue
        normalized.append(
            {
                "id": str(checkpoint.get("id") or f"cp{index}"),
                "description": str(checkpoint.get("description") or ""),
                "required": bool(checkpoint.get("required", True)),
                "evidence_hint": str(checkpoint.get("evidence_hint") or ""),
                "checkpoint_type": _normalize_checkpoint_type(checkpoint.get("checkpoint_type")),
                "weight": _normalize_weight(checkpoint.get("weight")),
            }
        )
    if not normalized:
        normalized = [
            {
                "id": "cp1",
                "description": "The requested task goal is completed.",
                "required": True,
                "evidence_hint": "Look for final state evidence in trace screenshots.",
                "checkpoint_type": "outcome",
                "weight": 1.0,
            }
        ]
    standard_id = build_standard_id(task, goal)
    return {
        "format_version": CHECKPOINT_STANDARD_VERSION,
        "standard_id": standard_id,
        "goal_hash": hashlib.sha256(goal.encode("utf-8")).hexdigest()[:16],
        "task_name": task,
        "task_goal": goal,
        "task_goal_rewrite": str(parsed.get("task_goal_rewrite") or goal),
        "checkpoints": normalized,
        "success_rule": str(parsed.get("success_rule") or "All required checkpoints must pass."),
    }


def _normalize_retrieval(parsed: dict[str, Any]) -> dict[str, Any]:
    selected = []
    for value in parsed.get("selected_steps") or []:
        try:
            selected.append(int(value))
        except (TypeError, ValueError):
            continue
    selected = list(dict.fromkeys(selected))
    return {
        "selected_steps": selected,
        "trusted": bool(parsed.get("trusted", False)),
        "confidence": _clamp_float(parsed.get("confidence"), default=0.0),
        "rationale": str(parsed.get("rationale") or ""),
        "fallback_reason": str(parsed.get("fallback_reason") or ""),
    }


def _trusted_retrieval(retrieval: dict[str, Any], min_confidence: float) -> bool:
    return (
        bool(retrieval.get("trusted"))
        and _clamp_float(retrieval.get("confidence"), default=0.0) >= min_confidence
        and bool(retrieval.get("selected_steps"))
    )


def _choose_retrieval_for_judge(
    first_pass: dict[str, Any],
    repair_pass: dict[str, Any],
) -> dict[str, Any]:
    """Choose model-selected candidates without hard-failing on trust metadata."""
    if repair_pass.get("selected_steps"):
        return repair_pass
    return first_pass


def _with_neighbor_context(
    selected_numbers: list[int],
    steps: list[NormalizedStep],
) -> list[int]:
    """Add one-step local context around model-selected evidence steps."""
    available = {step.step for step in steps}
    expanded: set[int] = set()
    for number in selected_numbers:
        if not isinstance(number, int):
            continue
        for candidate in (number - 1, number, number + 1):
            if candidate in available:
                expanded.add(candidate)
    return sorted(expanded)


def _normalize_checkpoint_result(
    parsed: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    achieved = bool(parsed.get("achieved", False))
    score = _clamp_float(parsed.get("score"), default=1.0 if achieved else 0.0)
    return {
        "id": str(parsed.get("id") or checkpoint.get("id") or "cp1"),
        "achieved": achieved,
        "score": score,
        "confidence": _clamp_float(parsed.get("confidence"), default=0.0),
        "evidence": str(parsed.get("evidence") or ""),
        "missing_or_conflict": str(parsed.get("missing_or_conflict") or ""),
        "insufficient_trace": bool(parsed.get("insufficient_trace", False)),
        "needs_fallback_verification": bool(parsed.get("needs_fallback_verification", False)),
    }


def _normalize_read_requests(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    raw_requests = parsed.get("read_requests")
    if not isinstance(raw_requests, list):
        return normalized
    for item in raw_requests:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool:
            continue
        request = {
            "tool": tool,
            "reason": str(item.get("reason") or ""),
        }
        for key in ("path", "target", "root", "name"):
            if key in item and item.get(key) not in (None, ""):
                request[key] = str(item.get(key))
        normalized.append(request)
    return normalized


def _should_run_read_tools(
    result: dict[str, Any],
    *,
    record: NormalizedRecord,
    checkpoint: dict[str, Any],
    fallback_confidence_threshold: float,
) -> bool:
    if result.get("needs_fallback_verification") or result.get("insufficient_trace"):
        return True
    if _clamp_float(result.get("confidence"), default=0.0) < fallback_confidence_threshold:
        return True
    if result.get("achieved") is False and _persistent_checkpoint(record, checkpoint, result):
        return True
    return False


def _persistent_checkpoint(
    record: NormalizedRecord,
    checkpoint: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    text = "\n".join(
        str(value)
        for value in (
            record.task,
            record.goal,
            checkpoint.get("description"),
            checkpoint.get("evidence_hint"),
            result.get("evidence"),
            result.get("missing_or_conflict"),
        )
    ).lower()
    if not any(word in text for word in ("file", "folder", "directory", "note", "save", "move", "delete", "copy")):
        return False
    uncertainty_markers = (
        "missing",
        "insufficient",
        "no visible",
        "no visual",
        "not visible",
        "cannot",
        "uncertain",
        "stale",
        "still",
        "final screenshot",
        "confirmation",
    )
    return any(marker in text for marker in uncertainty_markers)


def _checkpoint_metadata(
    *,
    selected_steps: list[NormalizedStep],
    manifest: list[dict[str, Any]],
    retrieval: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evidence_steps": [step.step for step in selected_steps],
        "evidence_images": len(manifest),
        "retrieval_method": retrieval.get("method", "model_step_retrieval_v1"),
        "retrieval_trusted": bool(retrieval.get("trusted")),
        "retrieval_confidence": retrieval.get("confidence"),
        "retrieval_rationale": retrieval.get("rationale", ""),
        "retrieval_fallback_reason": retrieval.get("fallback_reason", ""),
        "model_read_requests": _normalize_read_requests(parsed),
        "message_strategy": "doubao_official_gui_history_with_screenshots_and_read_tools",
    }


def _stored_read_tool_results(record: NormalizedRecord) -> list[dict[str, Any]]:
    raw_evidence = record.raw.get("post_execution_evidence")
    if not isinstance(raw_evidence, list):
        return []
    normalized = []
    for index, item in enumerate(raw_evidence, start=1):
        if not isinstance(item, dict):
            continue
        normalized_item = {
            "type": item.get("type", "read_tool_result"),
            "index": item.get("index", index),
            "tool": item.get("tool") or item.get("name"),
            "status": item.get("status", "ok"),
            "request": item.get("request", {}),
            "command": item.get("command", ""),
            "output": item.get("output", ""),
            "source": "record_post_execution_evidence",
        }
        for key in (
            "returncode",
            "exists",
            "found",
            "matches",
            "interpretation",
            "screenshot_path",
            "final_ui",
        ):
            if key in item:
                normalized_item[key] = item[key]
        normalized.append(normalized_item)
    return normalized


def _compact_checkpoint_result_for_metadata(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "achieved": result.get("achieved"),
        "confidence": result.get("confidence"),
        "evidence": result.get("evidence"),
        "missing_or_conflict": result.get("missing_or_conflict"),
        "insufficient_trace": result.get("insufficient_trace"),
        "needs_fallback_verification": result.get("needs_fallback_verification"),
    }


def _retrieval_failure_result(
    checkpoint: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": str(checkpoint.get("id") or "cp1"),
        "achieved": False,
        "confidence": 0.0,
        "evidence": str(retrieval.get("rationale") or ""),
        "missing_or_conflict": str(
            retrieval.get("fallback_reason") or "Model retrieval was not trustworthy."
        ),
        "insufficient_trace": True,
        "evidence_steps": retrieval.get("selected_steps") or [],
        "evidence_images": 0,
        "retrieval_method": retrieval.get("method", "model_step_retrieval_untrusted"),
        "retrieval_trusted": False,
        "retrieval_confidence": retrieval.get("confidence", 0.0),
        "retrieval_rationale": str(retrieval.get("rationale") or ""),
        "retrieval_fallback_reason": str(retrieval.get("fallback_reason") or ""),
        "message_strategy": "doubao_official_gui_history_no_rule_fallback",
    }


def _aggregate_checkpoint_results(
    checkpoint_results: list[dict[str, Any]],
    standard: dict[str, Any],
) -> dict[str, Any]:
    checkpoints = [cp for cp in standard.get("checkpoints", []) if isinstance(cp, dict)]
    required_ids = {cp.get("id") for cp in checkpoints if cp.get("required", True)}
    by_id = {result.get("id"): result for result in checkpoint_results}
    required_results = [by_id.get(checkpoint_id) for checkpoint_id in required_ids]
    type_by_id = {str(cp.get("id")): str(cp.get("checkpoint_type") or "supporting") for cp in checkpoints}
    weight_by_id = {
        str(cp.get("id")): _effective_weight(
            _normalize_weight(cp.get("weight")),
            str(cp.get("checkpoint_type") or "supporting"),
        )
        for cp in checkpoints
    }
    weighted_total = 0.0
    weighted_max = 0.0
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint.get("id"))
        weight = weight_by_id.get(checkpoint_id, 0.0)
        result = by_id.get(checkpoint_id)
        result_score = _clamp_float((result or {}).get("score"), default=0.0)
        weighted_total += weight * result_score
        weighted_max += weight
    normalized_score = (weighted_total / weighted_max) if weighted_max > 0 else 0.0
    outcome_required = [
        cp for cp in checkpoints if cp.get("required", True) and cp.get("checkpoint_type") == "outcome"
    ]
    outcome_gate_passed = all(
        _clamp_float((by_id.get(str(cp.get("id"))) or {}).get("score"), default=0.0) >= 0.6
        for cp in outcome_required
    )
    predicted_reward = _predicted_reward_from_score(normalized_score, outcome_gate_passed=outcome_gate_passed)
    success = predicted_reward >= 0.5
    score = normalized_score
    missing = [
        checkpoint_id
        for checkpoint_id in required_ids
        if not (by_id.get(checkpoint_id) or {}).get("achieved")
    ]
    return {
        "checkpoint_results": checkpoint_results,
        "success": success,
        "completeness_score": score,
        "predicted_reward": predicted_reward,
        "outcome_gate_passed": outcome_gate_passed,
        "insufficient_trace": any(result.get("insufficient_trace") for result in checkpoint_results),
        "rationale": (
            "Checkpoint scoring indicates the task is sufficiently complete."
            if success
            else f"Missing required checkpoints under selected evidence: {', '.join(sorted(missing))}"
        ),
    }


def _steps_by_number(steps: list[NormalizedStep], numbers: list[int]) -> list[NormalizedStep]:
    by_number = {step.step: step for step in steps}
    selected: list[NormalizedStep] = []
    for number in numbers:
        step = by_number.get(number)
        if step and step not in selected:
            selected.append(step)
    return selected


def _sample_steps_for_retrieval(
    steps: list[NormalizedStep],
    *,
    limit: int,
) -> list[NormalizedStep]:
    """Sample long traces for retrieval prompt token control."""
    if len(steps) <= limit:
        return steps
    head = max(2, limit // 4)
    tail = max(2, limit // 4)
    middle_budget = max(0, limit - head - tail)
    sampled = steps[:head]
    if middle_budget > 0:
        middle = steps[head : len(steps) - tail]
        if middle:
            stride = max(1, len(middle) // middle_budget)
            sampled.extend(middle[::stride][:middle_budget])
    sampled.extend(steps[-tail:])
    deduped: list[NormalizedStep] = []
    seen: set[int] = set()
    for step in sampled:
        if step.step in seen:
            continue
        seen.add(step.step)
        deduped.append(step)
    return deduped


def _agreement_with_reward(eval_success: Any, official_success: Any) -> bool | None:
    if not isinstance(eval_success, bool) or official_success is None:
        return None
    return eval_success is bool(official_success)


def _agreement_with_reward_band(predicted_reward: Any, official_reward: Any) -> bool | None:
    if official_reward is None or predicted_reward is None:
        return None
    return _reward_band(predicted_reward) == _reward_band(official_reward)


def _reward_band(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed >= 0.75:
        return 1.0
    if parsed >= 0.25:
        return 0.5
    return 0.0


def _normalize_checkpoint_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"outcome", "consistency", "supporting"}:
        return raw
    return "supporting"


def _normalize_weight(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _effective_weight(weight: float, checkpoint_type: str) -> float:
    if weight > 0:
        return weight
    defaults = {
        "outcome": 1.0,
        "consistency": 0.7,
        "supporting": 0.4,
    }
    return defaults.get(checkpoint_type, 0.4)


def _predicted_reward_from_score(score: float, *, outcome_gate_passed: bool) -> float:
    if not outcome_gate_passed:
        return 0.0
    if score >= 0.85:
        return 1.0
    if score >= 0.45:
        return 0.5
    return 0.0


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _progress(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def _is_token_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "exceed max message tokens",
        "max message tokens",
        "total tokens of image and text exceed",
    )
    return any(marker in text for marker in markers)
