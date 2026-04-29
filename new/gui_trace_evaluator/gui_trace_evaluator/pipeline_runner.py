"""Run an external GUI task executor first, then evaluate its records."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from gui_trace_evaluator.record_adapter import load_records, normalize_record
from gui_trace_evaluator.runner import evaluate_records_file


def main() -> None:
    args = parse_args()
    records_path = Path(args.records)

    if args.execute:
        _run_execute_command(args)

    if not records_path.exists():
        raise SystemExit(f"Records file was not produced: {records_path}")

    records = load_records(records_path)
    if args.require_images:
        _require_images(records, records_path=records_path, image_root=args.image_root)

    if args.dry_run:
        print(f"Execution output records: {records_path}")
        print(f"Loaded records: {len(records)}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        default=None,
        help=(
            "External command that runs the GUI agent and writes --records. "
            "Use quotes, e.g. --execute \"python run_awe.py --output records.json\"."
        ),
    )
    parser.add_argument("--execute_cwd", default=None)
    parser.add_argument("--execute_timeout_seconds", type=int, default=None)
    parser.add_argument("--records", required=True, help="Records JSON produced by execution.")
    parser.add_argument("--output", default="results/eval.json")
    parser.add_argument("--checkpoint_dir", default="checkpoint_cache")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--require_images", action="store_true")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--regenerate_checkpoints", action="store_true")
    parser.add_argument("--require_reviewed_checkpoints", action="store_true")
    parser.add_argument("--provider", choices=["ark", "openai"], default="openai")
    parser.add_argument("--model", default="doubao-seed-1-8-251228")
    parser.add_argument("--base_url", default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="ARK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--timeout_seconds", type=int, default=180)
    parser.add_argument("--extra_body_json", default=None)
    parser.add_argument("--max_selected_steps", type=int, default=12)
    parser.add_argument("--max_screenshot_turns", type=int, default=5)
    parser.add_argument("--max_retrieval_trace_steps", type=int, default=40)
    parser.add_argument("--retrieval_min_confidence", type=float, default=0.55)
    parser.add_argument("--fallback_confidence_threshold", type=float, default=0.7)
    parser.add_argument("--image_resize_scale", type=float, default=0.5)
    parser.add_argument("--image_jpeg_quality", type=int, default=85)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable_read_tools", action="store_true")
    parser.add_argument("--adb_path", default=None)
    parser.add_argument("--adb_serial", default=None)
    parser.add_argument("--read_tool_timeout_seconds", type=int, default=10)
    return parser.parse_args()


def _run_execute_command(args: argparse.Namespace) -> None:
    command = shlex.split(args.execute)
    if not command:
        raise SystemExit("--execute cannot be empty.")
    print(f"Running execution command: {args.execute}")
    subprocess.run(
        command,
        cwd=args.execute_cwd,
        timeout=args.execute_timeout_seconds,
        check=True,
    )


def _require_images(
    records: list[dict],
    *,
    records_path: Path,
    image_root: str | None,
) -> None:
    missing = []
    for record in records:
        normalized = normalize_record(record, base_dir=records_path.parent, image_root=image_root)
        if not any(step.evidence_screenshot_path for step in normalized.steps):
            missing.append(normalized.task)
    if missing:
        raise SystemExit(
            "Records contain no usable screenshots for tasks: " + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
