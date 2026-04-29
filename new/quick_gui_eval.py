#!/usr/bin/env python3
"""Quickly run GUI-Demo AndroidWorld tasks and evaluate generated records."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "RetroSavePlaylist",
    "RecipeDeleteDuplicateRecipes3",
    "RecipeAddMultipleRecipesFromMarkor2",
    "MarkorMergeNotes",
    "MarkorCreateNoteAndSms",
)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    python = _python_path(root, args.python)
    output_root = _output_root(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.records:
        summaries = _evaluate_existing_records(
            args=args,
            root=root,
            python=python,
            output_root=output_root,
        )
    else:
        summaries = _run_tasks_and_eval(
            args=args,
            root=root,
            python=python,
            output_root=output_root,
        )

    _write_summary(output_root, summaries)
    print(f"\nSummary: {output_root / 'summary.json'}")
    print(f"Report:  {output_root / 'summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="android_world_clean repo root.",
    )
    parser.add_argument(
        "--records",
        default=None,
        help="Eval-only mode. Pass one results.json file or a directory to scan.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names for execute+evaluate mode.",
    )
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--python", default=None, help="Python interpreter. Defaults to .venv/bin/python.")
    parser.add_argument("--exec-model", default="doubao-seed-1-6-vision-250815")
    parser.add_argument("--eval-model", default="doubao-seed-1-8-251228")
    parser.add_argument("--provider", default="ark", choices=("ark", "openai"))
    parser.add_argument("--api-key-env", default="ARK_API_KEY")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--exec-timeout", type=int, default=900)
    parser.add_argument("--eval-timeout", type=int, default=900)
    parser.add_argument(
        "--task-timeout",
        action="append",
        default=[],
        help="Per-task timeout override, e.g. RetroSavePlaylist=240. Can repeat.",
    )
    parser.add_argument("--max-screenshot-history", type=int, default=5)
    parser.add_argument("--max-text-history-chars", type=int, default=6000)
    parser.add_argument("--max-selected-steps", type=int, default=12)
    parser.add_argument("--max-screenshot-turns", type=int, default=5)
    parser.add_argument("--max-retrieval-trace-steps", type=int, default=40)
    parser.add_argument("--image-resize-scale", type=float, default=0.5)
    parser.add_argument("--image-jpeg-quality", type=int, default=85)
    parser.add_argument("--no-eval", action="store_true", help="Only execute tasks; do not run evaluator.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def _evaluate_existing_records(
    *,
    args: argparse.Namespace,
    root: Path,
    python: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    records_path = Path(args.records).expanduser()
    if not records_path.is_absolute():
        records_path = root / records_path
    if records_path.is_dir():
        files = sorted(records_path.rglob("results.json"))
    else:
        files = [records_path]

    summaries: list[dict[str, Any]] = []
    for index, record_file in enumerate(files, start=1):
        label = record_file.parent.parent.name if record_file.parent.name == "exec" else record_file.parent.name
        eval_dir = output_root / f"eval_{index:02d}_{_safe_name(label)}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / "eval.json"
        log_path = eval_dir / "eval.log"
        print(f">>> EVAL {record_file}")
        rc, timed_out = _run_to_log(
            _eval_command(args, root, python, record_file, eval_path),
            log_path,
            timeout=args.eval_timeout,
            cwd=root,
            env=_env(root, args),
            dry_run=args.dry_run,
        )
        item = {
            "mode": "eval_only",
            "records": str(record_file),
            "eval_rc": rc,
            "eval_timeout": timed_out,
            "eval": str(eval_path) if eval_path.exists() else None,
            "log": str(log_path),
        }
        item.update(_read_eval_result(eval_path))
        summaries.append(item)
        print(f">>> DONE {record_file.name} rc={rc} timeout={timed_out}")
    return summaries


def _run_tasks_and_eval(
    *,
    args: argparse.Namespace,
    root: Path,
    python: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    timeouts = _task_timeouts(args.task_timeout)
    summaries: list[dict[str, Any]] = []

    for task in tasks:
        task_dir = output_root / f"{_safe_name(task)}_seed{args.seed}"
        exec_dir = task_dir / "exec"
        task_dir.mkdir(parents=True, exist_ok=True)
        record_path = exec_dir / "results.json"
        print(f"\n>>> EXEC {task}")
        rc, timed_out = _run_to_log(
            _exec_command(args, root, python, task, exec_dir),
            task_dir / "exec.log",
            timeout=timeouts.get(task, args.exec_timeout),
            cwd=root,
            env=_env(root, args),
            dry_run=args.dry_run,
        )
        item: dict[str, Any] = {
            "task": task,
            "seed": args.seed,
            "exec_rc": rc,
            "exec_timeout": timed_out,
            "dir": str(task_dir),
            "record": str(record_path) if record_path.exists() else None,
        }
        item.update(_read_record_result(record_path))
        print(f">>> EXEC_DONE {task} rc={rc} timeout={timed_out} record={record_path.exists()}")

        if record_path.exists() and not args.no_eval:
            eval_path = task_dir / "eval.json"
            print(f">>> EVAL {task}")
            erc, etimed_out = _run_to_log(
                _eval_command(args, root, python, record_path, eval_path),
                task_dir / "eval.log",
                timeout=args.eval_timeout,
                cwd=root,
                env=_env(root, args),
                dry_run=args.dry_run,
            )
            item.update(
                {
                    "eval_rc": erc,
                    "eval_timeout": etimed_out,
                    "eval": str(eval_path) if eval_path.exists() else None,
                }
            )
            item.update(_read_eval_result(eval_path))
            print(f">>> EVAL_DONE {task} rc={erc} timeout={etimed_out} eval={eval_path.exists()}")

        summaries.append(item)
        _write_summary(output_root, summaries)
        print(f">>> TASK_SUMMARY {json.dumps(item, ensure_ascii=False)}")
    return summaries


def _exec_command(
    args: argparse.Namespace,
    root: Path,
    python: Path,
    task: str,
    exec_dir: Path,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(root / "new/GUI-Demo/run_android_world_task.py"),
        "--task",
        task,
        "--seed",
        str(args.seed),
        "--output_dir",
        str(exec_dir),
        "--model",
        args.exec_model,
        "--max_steps",
        str(args.max_steps),
        "--max_screenshot_history",
        str(args.max_screenshot_history),
        "--max_text_history_chars",
        str(args.max_text_history_chars),
    ]


def _eval_command(
    args: argparse.Namespace,
    root: Path,
    python: Path,
    record_file: Path,
    eval_path: Path,
) -> list[str]:
    return [
        str(python),
        "-m",
        "gui_trace_evaluator.runner",
        "--records",
        str(record_file),
        "--output",
        str(eval_path),
        "--checkpoint_dir",
        str(root / "new/gui_trace_evaluator/checkpoints"),
        "--provider",
        args.provider,
        "--model",
        args.eval_model,
        "--api_key_env",
        args.api_key_env,
        "--max_selected_steps",
        str(args.max_selected_steps),
        "--max_screenshot_turns",
        str(args.max_screenshot_turns),
        "--max_retrieval_trace_steps",
        str(args.max_retrieval_trace_steps),
        "--image_resize_scale",
        str(args.image_resize_scale),
        "--image_jpeg_quality",
        str(args.image_jpeg_quality),
        "--resume",
    ]


def _run_to_log(
    cmd: list[str],
    log_path: Path,
    *,
    timeout: int,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = " ".join(cmd)
    if dry_run:
        print(command_line)
        log_path.write_text(command_line + "\n", encoding="utf-8")
        return 0, False
    with log_path.open("w", encoding="utf-8") as log:
        log.write(command_line + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.write(f"\n[TIMEOUT] killed after {timeout}s\n")
            return 124, True


def _read_record_result(record_path: Path) -> dict[str, Any]:
    if not record_path.exists():
        return {}
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))[0]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"record_read_error": str(exc)}
    return {
        "official_success": record.get("success"),
        "official_reward": record.get("reward"),
        "steps": record.get("steps"),
        "abort_reason": record.get("abort_reason"),
        "goal": record.get("goal_used") or record.get("goal"),
    }


def _read_eval_result(eval_path: Path) -> dict[str, Any]:
    if not eval_path.exists():
        return {}
    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        evaluation = payload["evaluations"][0]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"eval_read_error": str(exc)}
    return {
        "eval_success": evaluation.get("success"),
        "eval_score": evaluation.get("completeness_score"),
        "agreement": evaluation.get("agreement_with_reward"),
        "eval_rationale": evaluation.get("rationale"),
    }


def _write_summary(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    (output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 快速执行与评测结果",
        "",
        "| Task | Exec | Official | Eval | Agreement | Score | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in summaries:
        task = item.get("task") or item.get("records", "")
        exec_status = "timeout" if item.get("exec_timeout") else str(item.get("exec_rc", ""))
        official = f"{item.get('official_success')} / {item.get('official_reward')}"
        eval_status = str(item.get("eval_success", ""))
        agreement = str(item.get("agreement", ""))
        score = str(item.get("eval_score", ""))
        notes = item.get("abort_reason") or item.get("eval_rationale") or item.get("record_read_error") or ""
        lines.append(
            f"| {task} | {exec_status} | {official} | {eval_status} | {agreement} | {score} | {_md_cell(str(notes))} |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_root(root: Path, output_root: str | None) -> Path:
    if output_root:
        path = Path(output_root).expanduser()
        return path if path.is_absolute() else root / path
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "new/gui_trace_evaluator/results" / f"quick_gui_eval_{stamp}"


def _python_path(root: Path, value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path
    venv_python = root / ".venv/bin/python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def _env(root: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    sdk_roots = [
        env.get("ANDROID_HOME"),
        env.get("ANDROID_SDK_ROOT"),
        str(Path.home() / "Library/Android/sdk"),
    ]
    sdk_paths: list[str] = []
    for sdk_root in sdk_roots:
        if not sdk_root:
            continue
        sdk = Path(sdk_root).expanduser()
        sdk_paths.extend([str(sdk / "platform-tools"), str(sdk / "emulator")])
    paths = [
        *sdk_paths,
        "/opt/homebrew/bin",
        "/usr/local/bin",
        env.get("PATH", ""),
    ]
    env["PATH"] = ":".join(path for path in paths if path)
    env["PYTHONPATH"] = str(root / "new/gui_trace_evaluator")
    return env


def _task_timeouts(items: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid --task-timeout {item!r}; expected Task=seconds.")
        result[name.strip()] = int(value.strip())
    return result


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120]


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")[:300]


if __name__ == "__main__":
    main()
