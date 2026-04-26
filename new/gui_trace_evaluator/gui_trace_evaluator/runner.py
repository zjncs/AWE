"""CLI for the standalone Doubao GUI trace evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gui_trace_evaluator.evaluator import TraceEvaluator
from gui_trace_evaluator.models import ArkChatModel, OpenAICompatibleChatModel
from gui_trace_evaluator.read_tools import ReadToolConfig
from gui_trace_evaluator.record_adapter import load_records, normalize_record
from gui_trace_evaluator.statistics import compute_batch_statistics


def main() -> None:
    args = parse_args()
    records_path = Path(args.records)
    records = load_records(records_path)
    if args.max_records is not None:
        records = records[: args.max_records]

    if args.dry_run:
        _print_dry_run(records, base_dir=records_path.parent, image_root=args.image_root)
        return

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"No API key provided and {args.api_key_env} is not set.")

    output_path = evaluate_records_file(
        records_path=records_path,
        output_path=Path(args.output),
        model_args=args,
        api_key=api_key,
        records=records,
    )
    print(f"JSON: {output_path}")
    print(f"Report: {output_path.with_suffix('.md')}")


def evaluate_records_file(
    *,
    records_path: Path,
    output_path: Path,
    model_args: argparse.Namespace,
    api_key: str,
    records: list[dict[str, Any]] | None = None,
) -> Path:
    """Evaluate an existing records file and write JSON/Markdown outputs."""
    records = load_records(records_path) if records is None else records
    if model_args.max_records is not None:
        records = records[: model_args.max_records]

    model = _build_model(model_args, api_key)
    evaluator = TraceEvaluator(
        model,
        checkpoint_dir=model_args.checkpoint_dir,
        image_root=model_args.image_root,
        regenerate_checkpoints=model_args.regenerate_checkpoints,
        max_selected_steps=model_args.max_selected_steps,
        max_screenshot_turns=model_args.max_screenshot_turns,
        retrieval_min_confidence=model_args.retrieval_min_confidence,
        fallback_confidence_threshold=model_args.fallback_confidence_threshold,
        read_tool_config=ReadToolConfig(
            enabled=not model_args.disable_read_tools,
            adb_path=model_args.adb_path,
            adb_serial=model_args.adb_serial,
            timeout_seconds=model_args.read_tool_timeout_seconds,
        ),
    )
    evaluations = evaluator.evaluate_records(records, base_dir=records_path.parent)
    stats = compute_batch_statistics(evaluations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(records_path),
        "model": model_args.model,
        "provider": model_args.provider,
        "num_records": len(records),
        "evaluations": evaluations,
        "statistics": stats,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_path.with_suffix(".md")
    report_path.write_text(_markdown_report(evaluations, stats), encoding="utf-8")

    print(f"Evaluated {len(evaluations)} records.")
    if stats["agreement_rate"] is not None:
        print(f"Agreement rate: {stats['agreement_rate']:.1%}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Input records JSON.")
    parser.add_argument("--output", default="results/eval.json")
    parser.add_argument("--checkpoint_dir", default="checkpoint_cache")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--regenerate_checkpoints", action="store_true")
    parser.add_argument("--provider", choices=["ark", "openai"], default="openai")
    parser.add_argument("--model", required=False, default="doubao-seed-1-8-251228")
    parser.add_argument("--base_url", default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="ARK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--timeout_seconds", type=int, default=180)
    parser.add_argument("--extra_body_json", default=None)
    parser.add_argument("--max_selected_steps", type=int, default=30)
    parser.add_argument("--max_screenshot_turns", type=int, default=10)
    parser.add_argument("--retrieval_min_confidence", type=float, default=0.55)
    parser.add_argument("--fallback_confidence_threshold", type=float, default=0.7)
    parser.add_argument("--disable_read_tools", action="store_true")
    parser.add_argument("--adb_path", default=None)
    parser.add_argument("--adb_serial", default=None)
    parser.add_argument("--read_tool_timeout_seconds", type=int, default=10)
    return parser.parse_args()


def _build_model(args: argparse.Namespace, api_key: str):
    if args.provider == "ark":
        return ArkChatModel(
            model_name=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
        )
    return OpenAICompatibleChatModel(
        model_name=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        extra_body=json.loads(args.extra_body_json) if args.extra_body_json else None,
    )


def _print_dry_run(
    records: list[dict[str, Any]],
    *,
    base_dir: Path,
    image_root: str | None,
) -> None:
    print(f"Loaded records: {len(records)}")
    for record in records[:20]:
        normalized = normalize_record(record, base_dir=base_dir, image_root=image_root)
        image_steps = sum(1 for step in normalized.steps if step.evidence_screenshot_path)
        print(
            f"- {normalized.task} "
            f"success={normalized.official_success} reward={normalized.official_reward} "
            f"steps={len(normalized.steps)} image_steps={image_steps}"
        )


def _markdown_report(evaluations: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    lines = [
        "# Doubao GUI Trace Evaluation",
        "",
        "## Summary",
        "",
        f"- Evaluated: {stats['total_evaluated']} / {stats['total_records']}",
        f"- Errors: {stats['total_errors']}",
        f"- Skipped: {stats['total_skipped']}",
    ]
    if stats["agreement_rate"] is not None:
        lines.append(f"- Agreement rate: {stats['agreement_rate']:.1%}")
    lines.extend(
        [
            "",
            "## Details",
            "",
            "| Task | Status | Eval Success | Official Success | Score | Agreement |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in evaluations:
        lines.append(
            "| {task} | {status} | {success} | {official} | {score} | {agree} |".format(
                task=item.get("task"),
                status=item.get("status"),
                success=item.get("success"),
                official=item.get("official_success"),
                score=item.get("completeness_score"),
                agree=item.get("agreement_with_reward"),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
