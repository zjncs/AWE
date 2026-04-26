"""Volcengine/Doubao official-style GUI message construction."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from gui_trace_evaluator.record_adapter import NormalizedStep


IMAGE_RESIZE_SCALE = 0.5
IMAGE_JPEG_QUALITY = 85
DEFAULT_MAX_SCREENSHOT_TURNS = 5


PHONE_USE_DOUBAO_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots.
You need to inspect the history and complete the requested analysis.

## Output Format
Thought: ...
Action: ...

## Action Space
click(point='<point>x y</point>')
long_press(point='<point>x y</point>')
type(content='')
scroll(point='<point>x y</point>', direction='down or up or right or left')
open_app(app_name='')
drag(start_point='<point>x y</point>', end_point='<point>x y</point>')
press_home()
press_back()
finished(content='xxx')

## Note
- Use {language} in the Thought part.
- Keep actions grounded in the screenshot and action history.

## User Instruction
{instruction}
"""


def build_system_prompt(*, instruction: str, language: str = "Chinese") -> str:
    """Build the first user message following the official GUI prompt style."""
    return PHONE_USE_DOUBAO_TEMPLATE.format(instruction=instruction, language=language)


def build_trace_messages(
    *,
    instruction: str,
    steps: list[NormalizedStep],
    final_request: str,
    image_step_numbers: set[int] | None = None,
    max_screenshot_turns: int = DEFAULT_MAX_SCREENSHOT_TURNS,
    language: str = "Chinese",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build official-style messages and an image manifest.

    The structure mirrors the official GUI examples:
    first a user prompt, then assistant Thought/Action history, with selected
    user image messages interleaved. At most `max_screenshot_turns`
    image-bearing steps are attached, sampled across the selected evidence.
    """
    selected = image_step_numbers or set()
    candidate_image_steps = [
        step
        for step in steps
        if step.step in selected and step.evidence_screenshot_path
    ]
    image_steps = _select_evenly_spaced_steps(
        candidate_image_steps,
        limit=max_screenshot_turns,
    )
    image_step_set = {step.step for step in image_steps}

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_system_prompt(instruction=instruction, language=language)}
    ]
    manifest: list[dict[str, Any]] = []
    for step in steps:
        if step.step in image_step_set:
            manifest.append(
                {
                    "image_index": len(manifest) + 1,
                    "step": step.step,
                    "kind": step.evidence_screenshot_kind,
                    "path": step.evidence_screenshot_path,
                }
            )
            if step.evidence_screenshot_kind == "before":
                messages.append(_step_screenshot_message(step))
        messages.append({"role": "assistant", "content": step_as_assistant_message(step)})
        if step.step in image_step_set and step.evidence_screenshot_kind == "after":
            messages.append(_step_screenshot_message(step))
    messages.append({"role": "user", "content": final_request})
    return messages, manifest


def _select_evenly_spaced_steps(steps: list[NormalizedStep], *, limit: int) -> list[NormalizedStep]:
    if limit <= 0:
        return []
    if len(steps) <= limit:
        return steps
    if limit == 1:
        return [steps[-1]]
    last_index = len(steps) - 1
    indices = [
        int(index * last_index / (limit - 1))
        for index in range(limit)
    ]
    return [steps[index] for index in dict.fromkeys(indices)]


def _step_screenshot_message(step: NormalizedStep) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Screenshot for step {step.step} "
                    f"({step.evidence_screenshot_kind} action)."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(step.evidence_screenshot_path)},
            },
        ],
    }


def step_as_assistant_message(step: NormalizedStep) -> str:
    """Render one trace step as official-style Thought/Action history."""
    thought = step.thinking or step.summary or "No explicit thought was recorded."
    action = step.action or "unknown_action()"
    lines = [f"Thought: {thought}", f"Action: {action}"]
    if step.summary:
        lines.append(f"Observation Summary: {step.summary}")
    before_ui = _compact_ui_text(step.before_ui)
    after_ui = _compact_ui_text(step.after_ui)
    if before_ui:
        lines.append(f"UI Text Before:\n{before_ui}")
    if after_ui:
        lines.append(f"UI Text After:\n{after_ui}")
    return "\n".join(lines)


def _compact_ui_text(value: str, *, max_lines: int = 200, max_chars: int = 15000) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines[:max_lines])
    if len(compact) > max_chars:
        compact = compact[: max_chars - 13].rstrip() + "\n...[truncated]"
    elif len(lines) > max_lines:
        compact = compact.rstrip() + "\n...[truncated]"
    return compact


def image_path_to_data_url(image_path: str) -> str:
    """Encode a screenshot as compressed data URL for multimodal messages."""
    path = Path(image_path)
    mime_type, image_bytes = _compressed_image_bytes(path)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _compressed_image_bytes(path: Path) -> tuple[str, bytes]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            target_size = (
                max(1, int(width * IMAGE_RESIZE_SCALE)),
                max(1, int(height * IMAGE_RESIZE_SCALE)),
            )
            if target_size != image.size:
                image = image.resize(target_size)
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
            return "image/jpeg", output.getvalue()
    except Exception:
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return mime_type, path.read_bytes()
