#!/usr/bin/env python3
"""Run the checkpoint-based trace evaluator over AndroidWorld results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from awe.evaluator import _atomic_write_text
from awe.evaluator import TraceEvaluator
from awe.models import OpenAIChatModel
from awe.presets import get_sample_probe_records, preset_task_names
from awe.statistics import compute_batch_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_path", default=None, help="Path to results.json.")
    parser.add_argument(
        "--sample_probe",
        action="store_true",
        help="Use built-in synthetic baseline traces instead of --results_path.",
    )
    parser.add_argument("--preset", choices=["baseline_probe"], default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task names.")
    parser.add_argument("--granularities", default=None, help="Comma-separated levels.")
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--checkpoint_dir",
        default=str(PROJECT_ROOT / "awe" / "checkpoint_cache"),
    )
    parser.add_argument("--regenerate_checkpoints", action="store_true")
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--base_url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--request_timeout_seconds", type=int, default=180)
    parser.add_argument(
        "--extra_body_json",
        default=None,
        help='Optional OpenAI extra_body JSON, e.g. {"thinking":{"type":"disabled"}}.',
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only load/filter records and report trace availability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _filter_records(_load_records(args), args)

    if args.dry_run:
        _print_dry_run(records)
        return

    if not args.model_name:
        raise SystemExit("--model_name is required unless --dry_run is used.")
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"No API key provided and {args.api_key_env} is not set.")

    evaluator = TraceEvaluator(
        OpenAIChatModel(
            model_name=args.model_name,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout_seconds,
            extra_body=json.loads(args.extra_body_json) if args.extra_body_json else None,
        ),
        checkpoint_dir=args.checkpoint_dir,
        regenerate_checkpoints=args.regenerate_checkpoints,
    )
    evaluations = evaluator.evaluate_records(records)
    stats = compute_batch_statistics(evaluations)

    output_path = _output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "sample_probe" if args.sample_probe else args.results_path,
        "model": args.model_name,
        "num_records": len(records),
        "evaluations": evaluations,
        "statistics": stats,
    }
    _atomic_write_text(
        output_path,
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = output_path.with_suffix(".md")
    _atomic_write_text(
        report_path,
        _markdown_report(evaluations, stats),
        encoding="utf-8",
    )

    print(f"Evaluated {len(evaluations)} records.")
    if stats["agreement_rate"] is not None:
        print(f"Agreement rate: {stats['agreement_rate']:.1%}")
    print(f"JSON: {output_path}")
    print(f"Report: {report_path}")


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_probe:
        return get_sample_probe_records()
    if not args.results_path:
        raise SystemExit("Provide --results_path or --sample_probe.")
    data = json.loads(Path(args.results_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Expected results_path to contain a JSON list.")
    return data


def _filter_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    task_names: set[str] | None = None
    if args.preset:
        task_names = set(preset_task_names(args.preset))
    if args.tasks:
        explicit = {task.strip() for task in args.tasks.split(",") if task.strip()}
        task_names = explicit if task_names is None else task_names & explicit

    granularities = (
        {item.strip() for item in args.granularities.split(",") if item.strip()}
        if args.granularities
        else None
    )
    filtered = []
    for record in records:
        if task_names is not None and record.get("task") not in task_names:
            continue
        if granularities is not None and record.get("granularity") not in granularities:
            continue
        filtered.append(record)
        if args.max_trials is not None and len(filtered) >= args.max_trials:
            break
    return filtered


def _print_dry_run(records: list[dict[str, Any]]) -> None:
    with_trace = sum(bool((record.get("trace") or {}).get("steps")) for record in records)
    print(f"Loaded {len(records)} records.")
    print(f"Records with compact trace: {with_trace}")
    print(f"Records without compact trace: {len(records) - with_trace}")
    for record in records[:20]:
        trace_steps = len((record.get("trace") or {}).get("steps") or [])
        print(
            f"- {record.get('task')} / {record.get('granularity')} "
            f"reward={record.get('reward')} trace_steps={trace_steps}"
        )


def _output_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if args.sample_probe:
        return PROJECT_ROOT / "awe" / "results" / "sample_probe_eval.json"
    return Path(args.results_path).with_name("llm_trace_eval.json")


def _markdown_report(
    evaluations: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    lines = [
        "# LLM Trace Evaluation",
        "",
        "## Summary",
        "",
        f"- **Evaluated**: {stats['total_evaluated']} / {stats['total_records']}",
        f"- **Errors**: {stats['total_errors']}",
        f"- **Skipped (no trace)**: {stats['total_skipped']}",
    ]
    if stats["agreement_rate"] is not None:
        lines.append(f"- **Agreement rate**: {stats['agreement_rate']:.1%}")
    lines.append("")

    # Confusion matrix
    cm = stats.get("confusion_matrix", {})
    lines.extend([
        "## Confusion Matrix (eval vs official)",
        "",
        "|  | Official Success | Official Fail |",
        "|--|------------------|---------------|",
        f"| **Eval Success** | {cm.get('true_positive', 0)} | {cm.get('false_positive', 0)} |",
        f"| **Eval Fail** | {cm.get('false_negative', 0)} | {cm.get('true_negative', 0)} |",
        "",
    ])

    # Per-task breakdown
    by_task = stats.get("by_task", {})
    if by_task:
        lines.append("## Per-Task Breakdown")
        lines.append("")
        lines.append("| Task | Count | Eval Success | Official Success | Agreement |")
        lines.append("|------|-------|-------------|------------------|-----------|")
        for task, data in by_task.items():
            agree_pct = data["agree"] / data["count"] if data["count"] else 0
            lines.append(
                f"| {task} | {data['count']} | {data['eval_success']} "
                f"| {data['official_success']} | {agree_pct:.0%} |"
            )
        lines.append("")

    # Detail table
    lines.extend([
        "## Detail Table",
        "",
        "| Task | Granularity | Status | Success | Score | Standard | Reward Agree |",
        "|------|-------------|--------|---------|-------|----------|--------------|",
    ])
    for item in evaluations:
        lines.append(
            "| {task} | {granularity} | {status} | {success} | {score} | {standard} | {agree} |".format(
                task=item.get("task", ""),
                granularity=item.get("granularity", ""),
                status=item.get("status", ""),
                success=item.get("success", ""),
                score=item.get("completeness_score", ""),
                standard=item.get("standard_id", ""),
                agree=item.get("agreement_with_reward", ""),
            )
        )
    lines.append("")

    # Detailed results per record
    lines.append("## Detailed Checkpoint Results")
    lines.append("")
    for item in evaluations:
        lines.append(f"### {item.get('task')} / {item.get('granularity')}")
        lines.append("")
        lines.append(f"- Status: `{item.get('status')}`")
        lines.append(f"- Standard: `{item.get('standard_id')}`")
        lines.append(f"- Success: `{item.get('success')}`")
        lines.append(f"- Score: `{item.get('completeness_score')}`")
        lines.append(f"- Rationale: {item.get('rationale', '')}")
        for checkpoint in item.get("checkpoint_results", []):
            mark = "+" if checkpoint.get("achieved") else "-"
            lines.append(
                f"  - [{mark}] {checkpoint.get('id')}: "
                f"conf={checkpoint.get('confidence')} | {checkpoint.get('evidence', '')}"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
