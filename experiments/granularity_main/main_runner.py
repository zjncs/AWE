#!/usr/bin/env python3
"""Formal main study runner — separate from ``pilot_granularity``.

Writes only under ``experiments/granularity_main/results/`` with run ids prefixed
``main_``.

Example:
  cd /path/to/android_world_clean
  export OPENAI_API_KEY=...
  python -m experiments.granularity_main.main_runner \\
      --model_name your-model-id \\
      --agent t3a \\
      --tasks ClockTimerEntry,ExpenseAddSingle
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from android_world import registry
from android_world.agents import infer
from android_world.env import env_launcher

from experiments.granularity_main.clean_prompts import (
    GRANULARITY_LEVELS,
    build_agent_goal,
    granularity_suffix,
)
from experiments.granularity_main.prompt_specs import CANONICAL_SPECS
from experiments.granularity_main.perturbations import (
    PerturbationContext,
    PerturbationKind,
    apply_perturbation_to_goal,
)
from experiments.granularity_main.task_set import MAIN_SIX

DEFAULT_MAX_STEPS = 999


def _find_adb() -> str:
    for p in (
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
    ):
        if os.path.isfile(p):
            return p
    raise EnvironmentError("adb not found")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--agent", choices=["m3a", "t3a"], default="t3a")
    ap.add_argument(
        "--base_url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    ap.add_argument(
        "--api_key",
        default=None,
        help="If omitted, uses OPENAI_API_KEY from the environment.",
    )
    ap.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task class names (default: formal MAIN_SIX).",
    )
    ap.add_argument(
        "--levels",
        default=None,
        help="Comma-separated granularity levels (default: intent,workflow,action).",
    )
    ap.add_argument("--console_port", type=int, default=5554)
    ap.add_argument("--adb_path", default=None)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--request_timeout_seconds", type=int, default=180)
    ap.add_argument("--max_retry", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--perturbation",
        choices=[k.value for k in PerturbationKind],
        default=PerturbationKind.NONE.value,
        help="Study B hook; default is clean (none).",
    )
    return ap.parse_args()


def _parse_levels(raw_levels: str | None) -> list[str]:
    levels = (
        [x.strip() for x in raw_levels.split(",") if x.strip()]
        if raw_levels
        else list(GRANULARITY_LEVELS)
    )
    invalid = [level for level in levels if level not in GRANULARITY_LEVELS]
    if invalid:
        raise ValueError(
            f"Unsupported granularity level(s): {', '.join(invalid)}. "
            f"Expected a subset of {', '.join(GRANULARITY_LEVELS)}."
        )
    return levels


def _trial_identity(
    task: str,
    granularity: str,
    seed: int,
    model: str,
    agent: str,
    temperature: float | None,
    perturbation: str,
) -> tuple[str, str, int, str, str, float | None, str]:
    return (task, granularity, seed, model, agent, temperature, perturbation)


def _validate_resume_compatibility(
    results: list[dict[str, Any]],
    *,
    model_name: str,
    agent: str,
    temperature: float,
    perturbation: str,
) -> None:
    mismatches: list[str] = []
    for r in results:
        if "error" in r:
            continue
        observed = {
            "model": r.get("model"),
            "agent": r.get("agent"),
            "temperature": r.get("temperature"),
            "perturbation": r.get("perturbation", PerturbationKind.NONE.value),
        }
        expected = {
            "model": model_name,
            "agent": agent,
            "temperature": temperature,
            "perturbation": perturbation,
        }
        bad = [k for k, v in observed.items() if v != expected[k]]
        if bad:
            mismatches.append(
                f"{r.get('task')}/{r.get('granularity')}/seed={r.get('seed')} "
                f"has {observed}, expected {expected}"
            )
    if mismatches:
        details = "\n".join(mismatches[:5])
        more = "" if len(mismatches) <= 5 else f"\n... and {len(mismatches) - 5} more"
        raise ValueError(
            "Resume file contains trials from a different run configuration.\n"
            f"{details}{more}"
        )


class TokenTracker:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
        }


def build_wrapper(args: argparse.Namespace) -> tuple[infer.OpenAICompatibleWrapper, TokenTracker]:
    kwargs: dict[str, Any] = {
        "model_name": args.model_name,
        "api_base_url": args.base_url,
        "temperature": args.temperature,
        "request_timeout_seconds": args.request_timeout_seconds,
        "max_retry": args.max_retry,
    }
    if args.api_key:
        kwargs["api_key"] = args.api_key
    wrapper = infer.OpenAICompatibleWrapper(**kwargs)
    tracker = TokenTracker()

    _orig_predict_mm = wrapper.predict_mm

    def _tracked_predict_mm(text_prompt, images=None):
        if images is None:
            images = []
        result = _orig_predict_mm(text_prompt, images)
        resp = result[2]
        if hasattr(resp, "usage") and resp.usage:
            tracker.prompt_tokens += resp.usage.prompt_tokens or 0
            tracker.completion_tokens += resp.usage.completion_tokens or 0
        tracker.calls += 1
        return result

    wrapper.predict_mm = _tracked_predict_mm
    wrapper.predict = lambda text_prompt: _tracked_predict_mm(text_prompt, [])
    return wrapper, tracker


def _make_agent(agent_type: str, env, wrapper):
    if agent_type == "m3a":
        # Lazy import to avoid heavy multimodal deps unless needed.
        from android_world.agents import m3a  # type: ignore

        return m3a.M3A(env, wrapper)
    from android_world.agents import t3a  # type: ignore

    return t3a.T3A(env, wrapper)


def run_single_trial(
    env,
    wrapper: infer.OpenAICompatibleWrapper,
    tracker: TokenTracker,
    agent_type: str,
    task_class_name: str,
    granularity: str,
    max_steps: int,
    seed: int,
    aw_registry: dict,
    perturbation: PerturbationKind,
) -> dict[str, Any]:
    if task_class_name not in aw_registry:
        return {
            "study": "granularity_main",
            "task": task_class_name,
            "granularity": granularity,
            "error": f"Task {task_class_name} not found in registry",
        }
    if task_class_name not in CANONICAL_SPECS:
        return {
            "study": "granularity_main",
            "task": task_class_name,
            "granularity": granularity,
            "error": f"No canonical spec in prompt_specs.CANONICAL_SPECS for {task_class_name}",
        }

    task_type = aw_registry[task_class_name]
    random.seed(seed)
    params = task_type.generate_random_params()
    task = task_type(params)

    base_goal = task.goal
    suffix = granularity_suffix(task_class_name, granularity, params)
    goal = build_agent_goal(base_goal, task_class_name, granularity, params)

    pctx = PerturbationContext(
        task_class=task_class_name,
        granularity=granularity,
        seed=seed,
        kind=perturbation,
    )
    goal, pmeta = apply_perturbation_to_goal(goal, pctx)

    try:
        task.initialize_task(env)
    except Exception as exc:
        return {
            "study": "granularity_main",
            "task": task_class_name,
            "granularity": granularity,
            "error": f"initialize_task failed: {exc}",
        }

    agent = _make_agent(agent_type, env, wrapper)
    tracker.reset()

    print(f"\n{'='*60}")
    print(f"  [main] {task_class_name}  |  {granularity}  |  {agent_type}")
    print(f"  Goal: {goal[:160]}{'...' if len(goal) > 160 else ''}")
    print(f"{'='*60}")

    LOOP_WINDOW = 5
    t0 = time.time()
    is_done = False
    steps_taken = 0
    abort_reason = None
    recent_summaries: list[str] = []
    consecutive_errors = 0

    for step in range(max_steps):
        try:
            result = agent.step(goal)
            steps_taken = step + 1
            consecutive_errors = 0
            summary = ""
            if result.data:
                summary = str(result.data.get("summary", result.data.get("action", "")))
            recent_summaries.append(summary)
            print(f"  Step {steps_taken} done.")
            if result.done:
                is_done = True
                break
            if len(recent_summaries) >= LOOP_WINDOW:
                tail = recent_summaries[-LOOP_WINDOW:]
                if tail[0] and all(s == tail[0] for s in tail):
                    abort_reason = f"loop detected (same action repeated {LOOP_WINDOW} times)"
                    print(f"  ⚠️ Aborting: {abort_reason}")
                    break
        except Exception as exc:
            consecutive_errors += 1
            print(f"  Step {step + 1} error ({consecutive_errors}): {exc}")
            if consecutive_errors >= 3 or "401" in str(exc):
                abort_reason = f"repeated errors ({consecutive_errors})"
                print(f"  ⚠️ Aborting trial due to {abort_reason}.")
                break

    elapsed = time.time() - t0
    time.sleep(3)
    try:
        reward = task.is_successful(env)
    except Exception as exc:
        reward = None
        print(f"  is_successful error: {exc}")

    success = reward is not None and reward >= 0.5

    try:
        task.tear_down(env)
    except Exception:
        pass

    print(
        f"  → {'✅ Success' if success else '❌ Fail'}  "
        f"(reward={reward}, steps={steps_taken}, time={elapsed:.1f}s)"
    )

    return {
        "study": "granularity_main",
        "task": task_class_name,
        "granularity": granularity,
        "perturbation": perturbation.value,
        "perturbation_meta": pmeta,
        "base_goal": base_goal,
        "guidance_suffix": suffix,
        "goal_used": goal,
        "success": success,
        "reward": reward,
        "agent_done": is_done,
        "abort_reason": abort_reason,
        "steps": steps_taken,
        "max_steps": max_steps,
        "elapsed_seconds": round(elapsed, 2),
        "model": wrapper.model,
        "agent": agent_type,
        "temperature": getattr(wrapper, "temperature", None),
        "seed": seed,
        **tracker.snapshot(),
    }


def main() -> None:
    args = parse_args()
    perturbation = PerturbationKind(args.perturbation)

    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_names = list(MAIN_SIX)

    for t in task_names:
        if t not in CANONICAL_SPECS:
            print(f"Unknown task or missing canonical spec: {t}", file=sys.stderr)
            sys.exit(1)

    try:
        levels = _parse_levels(args.levels)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    model_short = (
        args.model_name.split("-")[1]
        if "-" in args.model_name
        else args.model_name.replace("/", "_")[:32]
    )
    run_id = f"main_{model_short}_{args.agent}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.resume:
        output_dir = Path(args.resume).parent
    else:
        output_dir = (
            PROJECT_ROOT / "experiments" / "granularity_main" / "results" / run_id
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    adb_path = args.adb_path or _find_adb()
    print(f"Connecting to emulator on port {args.console_port} ...")
    env = env_launcher.load_and_setup_env(
        console_port=args.console_port,
        emulator_setup=False,
        adb_path=adb_path,
    )
    env.reset(go_home=True)

    wrapper, tracker = build_wrapper(args)
    print(f"Model: {wrapper.model}  Agent: {args.agent}  Base URL: {wrapper.api_base_url}")

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    all_results: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, int, str, str, float | None, str]] = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            all_results = json.loads(resume_path.read_text())
            try:
                _validate_resume_compatibility(
                    all_results,
                    model_name=args.model_name,
                    agent=args.agent,
                    temperature=args.temperature,
                    perturbation=perturbation.value,
                )
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                sys.exit(1)
            for r in all_results:
                if "error" not in r:
                    completed_keys.add(
                        _trial_identity(
                            r["task"],
                            r["granularity"],
                            r.get("seed", args.seed),
                            r.get("model", args.model_name),
                            r.get("agent", args.agent),
                            r.get("temperature", args.temperature),
                            r.get("perturbation", PerturbationKind.NONE.value),
                        )
                    )
            print(
                f"Resumed {len(all_results)} existing results, "
                f"{len(completed_keys)} completed trials to skip."
            )

    for task_cls in task_names:
        for level in levels:
            trial_key = _trial_identity(
                task_cls,
                level,
                args.seed,
                args.model_name,
                args.agent,
                args.temperature,
                perturbation.value,
            )
            if trial_key in completed_keys:
                print(
                    f"  ⏭️  Skipping {task_cls}/{level}/seed={args.seed} (already completed)"
                )
                continue
            env.reset(go_home=True)
            trial = run_single_trial(
                env,
                wrapper,
                tracker,
                args.agent,
                task_cls,
                level,
                args.max_steps,
                args.seed,
                aw_registry,
                perturbation,
            )
            all_results.append(trial)
            results_file = output_dir / "results.json"
            results_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
            print(f"  [saved to {results_file}]")

    _write_markdown_report(all_results, output_dir, args, run_id, task_names, levels)

    print(f"\n{'='*70}")
    print("GRANULARITY MAIN — SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        mark = "✅" if r.get("success") else "❌"
        print(
            f"{r.get('task','?'):<40} {r.get('granularity','?'):<10} {mark}  "
            f"steps={r.get('steps','?')}  tokens={r.get('total_tokens','?')}"
        )
    print(f"\nResults: {output_dir}")
    env.close()


def _write_markdown_report(
    results: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    task_names: list[str],
    levels: list[str],
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Granularity main study — run report",
        "",
        f"- **Run id**: `{run_id}`",
        f"- **Date**: {ts}",
        f"- **Model**: `{args.model_name}`",
        f"- **Agent**: `{args.agent}`",
        f"- **Seed**: {args.seed}",
        f"- **Temperature**: {args.temperature}",
        f"- **Perturbation**: `{args.perturbation}`",
        f"- **Tasks**: {', '.join(task_names)}",
        f"- **Levels**: {', '.join(levels)}",
        "",
        "| Task | Granularity | Success | Reward | Steps | Tokens | Time (s) |",
        "|------|-------------|---------|--------|-------|--------|----------|",
    ]
    for r in results:
        mark = "✅" if r.get("success") else "❌"
        tok = f"{r.get('prompt_tokens', 0)}/{r.get('completion_tokens', 0)}"
        lines.append(
            f"| {r.get('task','')} | {r.get('granularity','')} | {mark} | "
            f"{r.get('reward','?')} | {r.get('steps','?')} | {tok} | "
            f"{r.get('elapsed_seconds','?')} |"
        )
    lines.append("")
    lines.append("## Trial goals (truncated)")
    for i, r in enumerate(results, 1):
        g = r.get("goal_used", "")
        preview = (g[:500] + "…") if len(g) > 500 else g
        lines.append(f"### Trial {i}: {r.get('task')} / {r.get('granularity')}")
        lines.append("")
        lines.append("```")
        lines.append(preview)
        lines.append("```")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [report: {report_path}]")


if __name__ == "__main__":
    main()
