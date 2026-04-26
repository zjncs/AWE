"""Run one AndroidWorld task with Doubao GUI actions and emit evaluator records."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from action_parser import parse_doubao_response
from android_world_adapter import create_task, find_adb, load_env, prepare_android_world_imports
from android_world_executor import AndroidWorldExecutor
from doubao_client import DEFAULT_BASE_URL, DEFAULT_EXECUTION_MODEL, DoubaoClient
from phone_prompt import HistoryTurn, build_step_messages
from record_writer import (
    action_summary,
    build_record,
    collect_post_execution_evidence,
    step_record,
    ui_to_text,
    write_records,
)
from screenshot_utils import save_state_screenshot


def main() -> None:
    args = parse_args()
    prepare_android_world_imports()
    output_dir = Path(args.output_dir)
    screenshot_dir = output_dir / "trace_images" / f"{args.task}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    adb_path = args.adb_path or find_adb()
    extra_body = _load_extra_body(args.extra_body_json, args.api_extra_body_json_env)

    client = DoubaoClient(
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        extra_body=extra_body,
    )

    env: Any | None = None
    executor: AndroidWorldExecutor | None = None
    task: Any | None = None
    goal_text = args.task
    steps: list[dict[str, Any]] = []
    turn_history: list[HistoryTurn] = []
    agent_done = False
    done_rejections = 0
    abort_reason = None
    reward = None
    success = None
    evidence: list[dict[str, Any]] = []
    started_at = time.time()

    try:
        print(f"Connecting to AndroidWorld emulator on port {args.console_port} ...")
        env, executor, task = _initialize_with_recovery(
            args=args,
            adb_path=adb_path,
        )
        goal = str(task.goal)
        goal_text = goal
        print(f"Task: {args.task}")
        print(f"Goal: {goal}")

        # Doubao 0..1000 normalized coords must map to the same pixel space ADB uses.
        # Use env.logical_screen_size (W,H), not the JPEG file size, in case they differ.
        screen_size = env.logical_screen_size

        # Post-transition state idiom: fetch the first state once; afterwards reuse
        # the post-execute state as the next step's before-state instead of a fresh get.
        before_state = _safe_get_state(
            env,
            wait_to_stabilize=True,
            timeout_seconds=args.state_timeout_seconds,
        )
        _log_screen_alignment_once(
            before_state=before_state, screen_size=screen_size, label="(step 1 before)"
        )

        for step_index in range(1, args.max_steps + 1):
            before_path = screenshot_dir / f"step_{step_index:03d}_before.jpg"
            save_state_screenshot(before_state, before_path)

            messages = build_step_messages(
                goal=goal,
                screenshot_path=str(before_path),
                history_turns=turn_history,
                max_screenshot_history=args.max_screenshot_history,
                current_ui_text="" if args.disable_ui_text else ui_to_text(before_state),
                language=args.language,
            )
            response = ""
            parsed = None
            last_step_error: Exception | None = None
            for attempt in range(1, args.llm_step_retries + 1):
                try:
                    response = client.complete(messages)
                    print(f"\n---------- step {step_index} (attempt {attempt}/{args.llm_step_retries})")
                    print(response)
                    parsed = parse_doubao_response(response)
                    break
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    last_step_error = exc
                    if attempt < args.llm_step_retries:
                        print(
                            f"[gui-demo] step {step_index} attempt {attempt} failed: {exc}; "
                            "retrying with the same state."
                        )
                        time.sleep(min(2 * attempt, 5))
            if parsed is None:
                exc = last_step_error or RuntimeError("unknown step generation failure")
                abort_reason = f"step {step_index} failed after retries: {exc}"
                after_state = _safe_get_state(
                    env,
                    wait_to_stabilize=False,
                    timeout_seconds=args.state_timeout_seconds,
                    fallback=before_state,
                )
                after_path = screenshot_dir / f"step_{step_index:03d}_after.jpg"
                save_state_screenshot(after_state, after_path)
                steps.append(
                    {
                        "step": step_index,
                        "thinking": "",
                        "thought": "",
                        "action": "",
                        "summary": abort_reason,
                        "raw_response": response,
                        "before_screenshot_path": str(before_path),
                        "after_screenshot_path": str(after_path),
                        "before_ui": "",
                        "after_ui": "",
                        "error": str(exc),
                    }
                )
                print(f"Abort: {abort_reason}")
                break

            try:
                json_action, done = _call_with_timeout(
                    lambda: executor.execute(
                        parsed, screen_size=screen_size, before_state=before_state
                    ),
                    timeout_seconds=args.action_timeout_seconds,
                    label=f"executor.execute(step={step_index})",
                )
                after_state = _safe_get_state(
                    env,
                    wait_to_stabilize=True,
                    timeout_seconds=args.state_timeout_seconds,
                )
                after_path = screenshot_dir / f"step_{step_index:03d}_after.jpg"
                save_state_screenshot(after_state, after_path)
                steps.append(
                    step_record(
                        step=step_index,
                        raw_response=response,
                        parsed_action=parsed,
                        json_action=json_action,
                        before_screenshot_path=str(before_path),
                        after_screenshot_path=str(after_path),
                        before_state=before_state,
                        after_state=after_state,
                        summary=action_summary(parsed, json_action),
                    )
                )
                turn_history.append(
                    HistoryTurn(
                        screenshot_path=str(before_path),
                        assistant_output=response,
                    )
                )
                before_state = after_state
            except Exception as exc:  # pylint: disable=broad-exception-caught
                abort_reason = f"step {step_index} failed: {exc}"
                after_state = _safe_get_state(
                    env,
                    wait_to_stabilize=False,
                    timeout_seconds=args.state_timeout_seconds,
                    fallback=before_state,
                )
                after_path = screenshot_dir / f"step_{step_index:03d}_after.jpg"
                save_state_screenshot(after_state, after_path)
                steps.append(
                    {
                        "step": step_index,
                        "thinking": "",
                        "thought": "",
                        "action": "",
                        "summary": abort_reason,
                        "raw_response": response,
                        "before_screenshot_path": str(before_path),
                        "after_screenshot_path": str(after_path),
                        "before_ui": "",
                        "after_ui": "",
                        "error": str(exc),
                    }
                )
                print(f"Abort: {abort_reason}")
                break

            if done:
                if _done_gate_passed(
                    task=task,
                    env=env,
                    timeout_seconds=args.done_check_timeout_seconds,
                ):
                    agent_done = True
                    print("Agent marked task complete (done gate passed).")
                    break
                done_rejections += 1
                print(
                    "[gui-demo] Agent marked complete, but done gate failed; "
                    f"continue exploring (rejection {done_rejections}/"
                    f"{args.max_done_rejections})."
                )
                if done_rejections >= args.max_done_rejections:
                    abort_reason = (
                        "agent repeatedly marked complete but done gate failed"
                    )
                    print(f"Abort: {abort_reason}")
                    break

        time.sleep(args.final_wait_seconds)
        final_state = _safe_get_state(
            env,
            wait_to_stabilize=True,
            timeout_seconds=args.state_timeout_seconds,
            fallback=before_state,
        )
        final_screenshot_path = screenshot_dir / "final_state.jpg"
        save_state_screenshot(final_state, final_screenshot_path)
        try:
            reward = task.is_successful(env)
            success = bool(reward is not None and reward >= 0.5)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            abort_reason = f"is_successful failed: {exc}"
            print(abort_reason)

        evidence = collect_post_execution_evidence(
            goal=goal_text,
            task_name=args.task,
            adb_path=adb_path,
            console_port=args.console_port,
            final_state=final_state,
            final_screenshot_path=str(final_screenshot_path),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        abort_reason = f"fatal_error: {exc}"
        print(abort_reason)
    finally:
        try:
            if task is not None and env is not None:
                task.tear_down(env)
        except Exception:
            pass
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

    record = build_record(
        task_name=args.task,
        goal=goal_text,
        seed=args.seed,
        steps=steps,
        reward=reward,
        success=success,
        agent_done=agent_done,
        abort_reason=abort_reason,
        elapsed_seconds=time.time() - started_at,
        post_execution_evidence=evidence,
        model=args.model,
    )
    output_path = write_records([record], output_dir / "results.json")
    print(f"Record written: {output_path}")
    print(f"Success={success} reward={reward} steps={len(steps)}")


def _load_extra_body(
    extra_body_json: str | None, env_var_name: str
) -> dict[str, Any] | None:
    raw = (extra_body_json or "").strip() or os.environ.get(env_var_name, "").strip()
    if not raw:
        return None
    return json.loads(raw)


def _safe_get_state(
    env: Any,
    *,
    wait_to_stabilize: bool,
    retries: int = 3,
    timeout_seconds: float = 20.0,
    fallback: Any | None = None,
) -> Any:
    """Best-effort state fetch to survive transient adb/env glitches."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _get_state_with_timeout(
                env,
                wait_to_stabilize=wait_to_stabilize,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_error = exc
            if attempt < retries:
                time.sleep(min(attempt, 2))
    if fallback is not None:
        return fallback
    raise RuntimeError(f"get_state failed after retries: {last_error}") from last_error


def _get_state_with_timeout(
    env: Any,
    *,
    wait_to_stabilize: bool,
    timeout_seconds: float,
) -> Any:
    """Protect env.get_state from indefinite hangs in env/a11y stack."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(env.get_state, wait_to_stabilize=wait_to_stabilize)
    try:
        return fut.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        # Cancellation is best-effort; do not wait for stuck worker.
        fut.cancel()
        raise TimeoutError(
            f"get_state timed out after {timeout_seconds:.1f}s "
            f"(wait_to_stabilize={wait_to_stabilize})"
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _done_gate_passed(
    *,
    task: Any,
    env: Any,
    timeout_seconds: float,
) -> bool:
    """Verify completion with official task signal before accepting finished()."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(task.is_successful, env)
    try:
        reward = fut.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        fut.cancel()
        return False
    except Exception:
        return False
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return bool(reward is not None and reward >= 0.5)


def _call_with_timeout(func: Any, *, timeout_seconds: float, label: str) -> Any:
    """Run a callable with hard timeout to avoid indefinite env/adb hangs."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(func)
    try:
        return fut.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        fut.cancel()
        raise TimeoutError(f"{label} timed out after {timeout_seconds:.1f}s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _initialize_with_recovery(*, args: argparse.Namespace, adb_path: str) -> tuple[Any, AndroidWorldExecutor, Any]:
    """Initialize env/task with bounded retries and ADB recovery."""
    last_error: Exception | None = None
    for attempt in range(1, args.setup_retries + 1):
        env: Any | None = None
        try:
            env = _call_with_timeout(
                lambda: load_env(
                    console_port=args.console_port,
                    adb_path=adb_path,
                    perform_emulator_setup=args.perform_emulator_setup,
                ),
                timeout_seconds=args.setup_timeout_seconds,
                label="load_env",
            )
            executor = AndroidWorldExecutor(env, transition_pause=args.transition_pause)
            task = _call_with_timeout(
                lambda: create_task(args.task, seed=args.seed),
                timeout_seconds=args.setup_timeout_seconds,
                label="create_task",
            )
            _call_with_timeout(
                lambda: env.reset(go_home=True),
                timeout_seconds=args.setup_timeout_seconds,
                label="env.reset",
            )
            # Match T3A/M3A: hide pointer overlay / automation UI so screenshots match.
            env.hide_automation_ui()
            _call_with_timeout(
                lambda: task.initialize_task(env),
                timeout_seconds=args.setup_timeout_seconds,
                label="task.initialize_task",
            )
            return env, executor, task
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_error = exc
            print(
                f"[gui-demo] setup attempt {attempt}/{args.setup_retries} failed: {exc}"
            )
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if attempt < args.setup_retries:
                _recover_adb_connection(
                    adb_path=adb_path,
                    console_port=args.console_port,
                    wait_seconds=args.recovery_wait_seconds,
                )
    raise RuntimeError(f"setup failed after retries: {last_error}") from last_error


def _recover_adb_connection(*, adb_path: str, console_port: int, wait_seconds: float) -> None:
    """Best-effort ADB recovery for transient env/a11y reset hangs."""
    serial = f"emulator-{console_port}"
    cmds = [
        [adb_path, "kill-server"],
        [adb_path, "start-server"],
        [adb_path, "-s", serial, "wait-for-device"],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=False, timeout=20, capture_output=True, text=True)
        except Exception:
            pass
    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _log_screen_alignment_once(
    *,
    before_state: Any,
    screen_size: tuple[int, int],
    label: str,
) -> None:
    """Warn if pixel buffer (W x H) differs from logical screen (for tap mapping)."""
    lw, lh = screen_size
    pixels = getattr(before_state, "pixels", None)
    if pixels is None:
        return
    arr = np.asarray(pixels)
    if arr.ndim < 2:
        return
    h, w = int(arr.shape[0]), int(arr.shape[1])
    print(f"[gui-demo] logical_screen_size {label}: {lw}x{lh}; pixel buffer: {w}x{h}")
    if (w, h) != (lw, lh):
        print(
            "[gui-demo] Note: 0..1000 is mapped with logical_screen_size; "
            "if taps look off, check display scaling or observation shape."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="FilesDeleteFile")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--output_dir", default="results/gui_demo_android_world")
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--adb_path", default=None)
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument("--model", default=DEFAULT_EXECUTION_MODEL)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="ARK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--timeout_seconds", type=int, default=120)
    parser.add_argument(
        "--llm_step_retries",
        type=int,
        default=4,
        help="Retries per step for empty/unparseable model outputs.",
    )
    parser.add_argument("--transition_pause", type=float, default=1.0)
    parser.add_argument("--final_wait_seconds", type=float, default=2.0)
    parser.add_argument(
        "--setup_timeout_seconds",
        type=float,
        default=60.0,
        help="Hard timeout for env.reset and task.initialize_task.",
    )
    parser.add_argument(
        "--setup_retries",
        type=int,
        default=3,
        help="Retries for environment/task setup before giving up.",
    )
    parser.add_argument(
        "--recovery_wait_seconds",
        type=float,
        default=2.0,
        help="Sleep after ADB recovery before next setup attempt.",
    )
    parser.add_argument(
        "--action_timeout_seconds",
        type=float,
        default=45.0,
        help="Hard timeout for each executor.execute call.",
    )
    parser.add_argument(
        "--state_timeout_seconds",
        type=float,
        default=20.0,
        help="Hard timeout per env.get_state call to avoid indefinite env hangs.",
    )
    parser.add_argument(
        "--done_check_timeout_seconds",
        type=float,
        default=10.0,
        help="Timeout for done-gate official success check.",
    )
    parser.add_argument(
        "--max_done_rejections",
        type=int,
        default=3,
        help="Abort after repeated finished() signals that fail done-gate check.",
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument(
        "--max_screenshot_history",
        type=int,
        default=5,
        help=(
            "Sliding window size for (screenshot, assistant) history turns attached "
            "to the Doubao chat. Matches the official Volc GUI demo default of 5."
        ),
    )
    parser.add_argument(
        "--disable_ui_text",
        action="store_true",
        help="Do not include Android accessibility UI text in the Doubao execution prompt.",
    )
    parser.add_argument(
        "--extra_body_json",
        default=None,
        help='JSON merged into Ark chat.completions, e.g. {"thinking":{"type":"disabled"}}',
    )
    parser.add_argument(
        "--api_extra_body_json_env",
        default="ARK_EXTRA_BODY_JSON",
        help="Env var to read JSON for extra_body if --extra_body_json is not set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
