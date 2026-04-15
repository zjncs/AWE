# Instruction Granularity — Formal Main Study

Working title: **Instruction Granularity Scaling Trends for GUI Agents**  
Subtitle (suggested): *Workflow as the Most Robust Default Prompt Granularity*

This directory holds the **frozen** main-study protocol, prompts, and results.  
Exploratory runs remain under `experiments/pilot_granularity/` and must not receive new formal main-study results.

## Formal task set (primary table)

| Task | Role |
|------|------|
| `ClockTimerEntry` | Low-complexity anchor; clean evaluator; good for over-specification cost. |
| `SimpleCalendarAddOneEvent` | High complexity: navigation + form. |
| `ExpenseAddSingle` | Structured multi-field form; balances workflow-heavy picks. |
| `FilesMoveFile` | Hierarchical navigation; subgoal structure. |
| `RecipeAddSingleRecipe` | Long structured input; tests intent under-constraint vs brittle action. |
| `SimpleSmsSendReceivedAddress` | Read-then-forward relay (not pure compose). |

**Backup (not in primary 6 unless substituted):** `MarkorMoveNote`

**Sanity / calibration only:** `ContactsAddContact`, `SimpleSmsSend` — use for intent-only ceiling/floor checks, not the main paper table.

**Excluded from formal primary results (for now):** `RetroCreatePlaylist`, `MarkorCreateNote`, Wi‑Fi–related tasks (`SystemWifiTurnOn`, `SystemWifiTurnOff`, `TurnOnWifiAndOpenApp`, `TurnOffWifiAndTurnOnBluetooth`) — unstable app/evaluator or known measurement issues.

Source of truth: `task_set.py`.

## Prompt fairness rules (fixed)

1. **Information equivalence** — Intent, workflow, and action describe the *same* task instance. Intent uses the official AndroidWorld `task.goal` only. Workflow and action **append** structured guidance derived from one canonical spec per task (`prompt_specs.py`), without adding new literal facts beyond what is already in `task.goal` (for tasks whose data lives entirely in `task.goal`, the appended blocks stay generic).
2. **Action** — UI-anchored natural-language steps only; **no screen coordinates**.
3. **Workflow** — 3–5 **sub-goals**, not click-by-click instructions.
4. **Perturbations** — Implemented in `perturbations.py`, versioned separately. Do not hand-edit clean prompts per trial.

Generation: `clean_prompts.py` builds the full agent goal from `task.goal` + optional suffix.

## Execution order (recommended)

1. **Freeze pilot** — Treat `pilot_granularity` as historical; no new formal aggregates there.
2. **Intent-only calibration** on the **primary 6** — strong + weak model, 2–3 seeds; confirm no universal ceiling/floor.
3. **Clean main experiment** — e.g. 2 models × 6 tasks × 3 granularities × 3 seeds = **108 trials** (T3A, temperature 0, fixed agent loop).
4. **Perturbed subset** — After clean results, add perturbations on the most discriminative **4** tasks (e.g. popup / `stale_action_step`); wire hooks in `perturbations.py` and set `--perturbation`.
5. **Claims** — Only after clean (+ optional perturbed) data, evaluate whether *workflow* is the most **robust default** granularity (not “always best”).

## Running

```bash
cd /path/to/android_world_clean
source .venv/bin/activate
export OPENAI_API_KEY=...
# optional: export OPENAI_BASE_URL=https://...

python -m experiments.granularity_main.main_runner \
  --model_name <model_id> \
  --agent t3a \
  --seed 30 \
  --temperature 0
```

- Note: importing `android_world.registry` may segfault under sandboxed execution in some IDE environments. If you see `exit code 139` before any emulator connection, re-run outside the sandbox.

- Default tasks: **MAIN_SIX** (see `task_set.py`). Override with `--tasks TaskA,TaskB`.
- Default levels: `intent`, `workflow`, `action`. Override with `--levels intent`.
- Invalid `--levels` values now fail fast instead of silently falling back to `action`.
- Results directory: `experiments/granularity_main/results/main_<model>_<agent>_<timestamp>/` (`results.json`, `report.md`).
- Resume: `--resume path/to/results.json` only for the **same** `model_name`, `agent`, `temperature`, and `perturbation`; mismatches fail fast.
- Perturbations: only `--perturbation none` is currently runnable. Non-clean perturbations intentionally fail fast until Study B hooks are implemented.

## Running On Another Machine

Minimum checklist:

1. Clone / fetch the handoff branch for this study fork.
2. Recreate the project virtualenv and install the same dependencies.
3. Ensure `adb` is available at `~/Library/Android/sdk/platform-tools/adb` or pass `--adb_path`.
4. Start the Android emulator on the target machine before launching the runner.
5. Export `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if needed).

Recommended first command on the new machine:

```bash
python -m experiments.granularity_main.main_runner \
  --model_name <model_id> \
  --agent t3a \
  --tasks ClockTimerEntry \
  --levels intent \
  --perturbation none \
  --seed 30
```

That sanity check confirms the emulator, API credentials, and runner wiring before you start a full matrix.

## File map

| File | Purpose |
|------|---------|
| `task_set.py` | Primary 6, backup, sanity list, exclusions. |
| `prompt_specs.py` | One `CanonicalTaskSpec` per supported task class. |
| `clean_prompts.py` | Intent/workflow/action suffixes from specs. |
| `perturbations.py` | Study B: popup, stale action, etc. (stubs until wired). |
| `main_runner.py` | Formal runner (isolated run-id scheme). |
| `results/` | Main-study outputs only. |

## Statistics (pointer)

For paper-grade analysis, prefer mixed models (e.g. logistic for success) with random effects for task and seed; plot clean vs perturbed success curves by model × granularity separately.
