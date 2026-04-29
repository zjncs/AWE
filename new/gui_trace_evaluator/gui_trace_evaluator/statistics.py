"""Batch statistics."""

from __future__ import annotations

from typing import Any


def compute_batch_statistics(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(evaluations)
    evaluated = [item for item in evaluations if item.get("status") == "evaluated"]
    errors = [item for item in evaluations if item.get("status") == "evaluation_error"]
    skipped = [item for item in evaluations if item.get("status") == "skipped_no_trace"]
    comparable = [item for item in evaluated if item.get("agreement_with_reward") is not None]
    agree = sum(1 for item in comparable if item.get("agreement_with_reward") is True)
    checkpoint_results = [
        checkpoint
        for item in evaluated
        for checkpoint in item.get("checkpoint_results", [])
        if isinstance(checkpoint, dict)
    ]
    read_tool_items = [
        checkpoint.get("read_tool_verification", {})
        for checkpoint in checkpoint_results
        if isinstance(checkpoint.get("read_tool_verification"), dict)
    ]
    read_tool_triggered = [
        item for item in read_tool_items if item.get("triggered") is True
    ]
    return {
        "total_records": total,
        "total_evaluated": len(evaluated),
        "total_errors": len(errors),
        "total_skipped": len(skipped),
        "agreement_rate": agree / len(comparable) if comparable else None,
        "step_count_buckets": _step_count_buckets(evaluated),
        "completeness_score": _score_summary(
            item.get("completeness_score") for item in evaluated
        ),
        "checkpoint_score": _score_summary(
            checkpoint.get("score") for checkpoint in checkpoint_results
        ),
        "read_tools": {
            "checkpoint_total": len(checkpoint_results),
            "triggered": len(read_tool_triggered),
            "trigger_rate": len(read_tool_triggered) / len(checkpoint_results)
            if checkpoint_results
            else None,
            "changed_to_success": sum(
                1
                for checkpoint in checkpoint_results
                if checkpoint.get("read_tool_verification", {})
                .get("first_pass_result", {})
                .get("achieved")
                is False
                and checkpoint.get("achieved") is True
            ),
        },
        "confusion_matrix": {
            "true_positive": sum(
                1 for item in comparable if item.get("success") is True and item.get("official_success") is True
            ),
            "false_positive": sum(
                1 for item in comparable if item.get("success") is True and item.get("official_success") is False
            ),
            "false_negative": sum(
                1 for item in comparable if item.get("success") is False and item.get("official_success") is True
            ),
            "true_negative": sum(
                1 for item in comparable if item.get("success") is False and item.get("official_success") is False
            ),
        },
    }


def _score_summary(values: Any) -> dict[str, Any]:
    nums = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
    ]
    if not nums:
        return {"count": 0, "avg": None, "min": None, "max": None}
    return {
        "count": len(nums),
        "avg": sum(nums) / len(nums),
        "min": min(nums),
        "max": max(nums),
    }


def _step_count_buckets(evaluated: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = {
        "short_0_5": [],
        "medium_6_15": [],
        "long_16_plus": [],
    }
    for item in evaluated:
        steps = item.get("trace_steps")
        if not isinstance(steps, int):
            continue
        if steps <= 5:
            key = "short_0_5"
        elif steps <= 15:
            key = "medium_6_15"
        else:
            key = "long_16_plus"
        buckets[key].append(item)
    return {
        key: {
            "count": len(items),
            "agreement_rate": _agreement_rate(items),
        }
        for key, items in buckets.items()
    }


def _agreement_rate(items: list[dict[str, Any]]) -> float | None:
    comparable = [item for item in items if item.get("agreement_with_reward") is not None]
    if not comparable:
        return None
    agree = sum(1 for item in comparable if item.get("agreement_with_reward") is True)
    return agree / len(comparable)
