"""Calibration helpers for evaluator outputs.

This does not change decisions. It summarizes how current confidence scores
correlate with AndroidWorld official success labels so thresholds can be chosen
from data instead of fixed by guesswork.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    evaluations = []
    for path in args.inputs:
        evaluations.extend(_load_evaluations(Path(path)))
    report = calibrate(evaluations)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evaluation JSON files.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def calibrate(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        item
        for item in evaluations
        if item.get("status") == "evaluated"
        and isinstance(item.get("official_success"), bool)
        and isinstance(item.get("success"), bool)
    ]
    return {
        "records": len(evaluated),
        "overall_agreement": _agreement(evaluated),
        "judge_confidence_thresholds": _threshold_sweep(evaluated, "judge_confidence"),
        "retrieval_confidence_thresholds": _threshold_sweep(evaluated, "retrieval_confidence"),
    }


def _load_evaluations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("evaluations"), list):
        return [item for item in data["evaluations"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported evaluation JSON shape: {path}")


def _agreement(items: list[dict[str, Any]]) -> float | None:
    if not items:
        return None
    return sum(item.get("success") is item.get("official_success") for item in items) / len(items)


def _threshold_sweep(evaluations: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    rows = []
    for threshold in [round(index / 20, 2) for index in range(0, 21)]:
        kept = [
            item
            for item in evaluations
            if _record_confidence(item, field) is not None
            and float(_record_confidence(item, field)) >= threshold
        ]
        rows.append(
            {
                "threshold": threshold,
                "coverage": len(kept) / len(evaluations) if evaluations else None,
                "agreement": _agreement(kept),
                "records": len(kept),
            }
        )
    return rows


def _record_confidence(item: dict[str, Any], field: str) -> float | None:
    values = []
    for checkpoint in item.get("checkpoint_results", []):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(field), (int, float)):
            values.append(float(checkpoint[field]))
    if not values:
        return None
    return min(values)


if __name__ == "__main__":
    main()
