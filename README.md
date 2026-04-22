# AWE — AndroidWorld Trace Evaluator

Checkpoint-based LLM trace evaluator for [AndroidWorld](https://github.com/google-research/android_world) execution traces.

## How it works

1. **Checkpoint Generation**: Given a task name and goal, an LLM generates a stable rubric (set of checkpoints) that defines what "success" looks like.
2. **Trace Evaluation**: The evaluator grades an execution trace against the checkpoint rubric, determining which checkpoints were achieved.
3. **Verdict**: If all *required* checkpoints are achieved, the trace is a success; otherwise it fails.

Checkpoints are cached by `(task_name, base_goal)` hash, so the same task always uses the same rubric across multiple runs.

## Install

```bash
pip install -e .
```

Requires Python 3.10+ and `openai>=1.0`.

## Quick Start

### Evaluate with built-in synthetic traces

```bash
export ARK_API_KEY="your-api-key"

python -m awe.runner \
    --sample_probe \
    --model_name your-model-id \
    --base_url https://your-api-endpoint/v1 \
    --api_key_env ARK_API_KEY
```

### Evaluate real execution results

```bash
python -m awe.runner \
    --results_path path/to/results.json \
    --model_name your-model-id \
    --base_url https://your-api-endpoint/v1 \
    --api_key_env ARK_API_KEY \
    --output_path eval_output
```

This produces:
- `eval_output.json` — Full evaluation results with per-checkpoint details
- `eval_output.md` — Human-readable report with confusion matrix

## Architecture

```
awe/
  __init__.py            # Package exports
  evaluator.py           # Core: checkpoint generation + trace evaluation
  models.py              # OpenAI-compatible LLM adapter with retry
  prompts.py             # Prompt templates for checkpoint gen & evaluation
  runner.py              # CLI entry point
  statistics.py          # Batch statistics (agreement rate, confusion matrix)
  json_utils.py          # Robust JSON parsing from LLM output
  presets.py             # Built-in synthetic test traces
  trace_serialization.py # Trace-to-prompt serialization with evidence selection
```

### Adaptive evaluation strategy

- **Short traces** (<=10 steps): Full-trace evaluation in a single LLM call. Best context preservation.
- **Long traces** (>10 steps): Per-checkpoint evaluation with relevance-filtered evidence. Scales to arbitrarily long traces.

### Checkpoint stability

Checkpoint rubrics are keyed by `SHA256(task_name + base_goal)`. Different granularity levels (intent / workflow / action) for the same task share the same rubric, ensuring consistent evaluation across prompt variants.

## Output format

The evaluator produces per-task results including:
- `success` — Whether all required checkpoints were achieved
- `completeness_score` — Fraction of required checkpoints achieved (0.0-1.0)
- `checkpoint_results` — Per-checkpoint detail with evidence and confidence
- `agreement_with_reward` — Whether eval verdict matches official reward

Batch statistics include:
- Agreement rate (eval vs official)
- Confusion matrix (TP / FP / FN / TN)
- Per-task and per-granularity breakdowns

## Tested models

- `doubao-seed-1-8-251228` (via Ark API) — 100% agreement on 7 test cases (4 synthetic + 3 real)
- Compatible with any OpenAI-compatible API endpoint

## License

MIT
