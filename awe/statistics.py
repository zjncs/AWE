"""Batch statistics for trace evaluation results."""

from __future__ import annotations

from typing import Any


def compute_batch_statistics(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes agreement rates, confusion matrix, and per-task breakdowns."""
    total = len(evaluations)
    evaluated = [e for e in evaluations if e.get("status") == "evaluated"]
    errors = [e for e in evaluations if e.get("status") == "evaluation_error"]
    skipped = [e for e in evaluations if e.get("status") == "skipped_no_trace"]

    # Only compare where both eval and official reward are available.
    comparable = [
        e for e in evaluated if e.get("agreement_with_reward") is not None
    ]
    agree_count = sum(1 for e in comparable if e["agreement_with_reward"] is True)

    # Confusion matrix: eval_success vs official_success
    tp = sum(
        1
        for e in comparable
        if e.get("success") is True and e.get("official_success") is True
    )
    fp = sum(
        1
        for e in comparable
        if e.get("success") is True and e.get("official_success") is False
    )
    fn = sum(
        1
        for e in comparable
        if e.get("success") is False and e.get("official_success") is True
    )
    tn = sum(
        1
        for e in comparable
        if e.get("success") is False and e.get("official_success") is False
    )

    # Per-task breakdown
    by_task: dict[str, dict[str, Any]] = {}
    for e in evaluated:
        task = e.get("task", "unknown")
        if task not in by_task:
            by_task[task] = {
                "count": 0,
                "eval_success": 0,
                "official_success": 0,
                "agree": 0,
            }
        entry = by_task[task]
        entry["count"] += 1
        if e.get("success"):
            entry["eval_success"] += 1
        if e.get("official_success"):
            entry["official_success"] += 1
        if e.get("agreement_with_reward") is True:
            entry["agree"] += 1

    # Per-granularity breakdown
    by_granularity: dict[str, dict[str, Any]] = {}
    for e in evaluated:
        gran = e.get("granularity", "unknown")
        if gran not in by_granularity:
            by_granularity[gran] = {"count": 0, "eval_success": 0, "agree": 0}
        entry = by_granularity[gran]
        entry["count"] += 1
        if e.get("success"):
            entry["eval_success"] += 1
        if e.get("agreement_with_reward") is True:
            entry["agree"] += 1

    # Checkpoint stability
    checkpoint_usage: dict[str, int] = {}
    for e in evaluated:
        sid = e.get("standard_id", "")
        checkpoint_usage[sid] = checkpoint_usage.get(sid, 0) + 1

    return {
        "total_records": total,
        "total_evaluated": len(evaluated),
        "total_errors": len(errors),
        "total_skipped": len(skipped),
        "agreement_rate": agree_count / len(comparable) if comparable else None,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "by_task": by_task,
        "by_granularity": by_granularity,
        "checkpoint_stability": {
            "unique_standards": len(checkpoint_usage),
            "max_reuse": max(checkpoint_usage.values()) if checkpoint_usage else 0,
        },
    }
