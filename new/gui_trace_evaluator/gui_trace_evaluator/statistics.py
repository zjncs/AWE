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
    return {
        "total_records": total,
        "total_evaluated": len(evaluated),
        "total_errors": len(errors),
        "total_skipped": len(skipped),
        "agreement_rate": agree / len(comparable) if comparable else None,
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
