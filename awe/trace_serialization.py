"""Converts agent step data into compact, JSON-safe execution traces."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


TRACE_FORMAT_VERSION = 2
DEFAULT_MAX_UI_ELEMENTS = 80
DEFAULT_MAX_TEXT_CHARS = 4000
DEFAULT_MAX_ELEMENT_CHARS = 500
DEFAULT_MAX_PROMPT_IMAGES = 12
DEFAULT_MAX_CHECKPOINT_EVIDENCE_STEPS = 12
DEFAULT_MAX_CHECKPOINT_IMAGES = 6
DEFAULT_EVIDENCE_NEIGHBOR_WINDOW = 1
DEFAULT_FINAL_STATE_TAIL_STEPS = 4
DEFAULT_MAX_CHECKPOINT_UI_ELEMENTS = 6
DEFAULT_MAX_CHECKPOINT_UI_FALLBACK = 2
DEFAULT_MAX_CHECKPOINT_REASON_CHARS = 400
DEFAULT_MAX_CHECKPOINT_SUMMARY_CHARS = 500
DEFAULT_MAX_CHECKPOINT_UI_CHARS = 220

_REASON_ACTION_RE = re.compile(r"Reason:\s*(.*?)\s*Action:\s*(.*)", re.DOTALL | re.I)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+./:-]{1,}")
_QUOTED_TEXT_RE = re.compile(r"['\"]([^'\"]{2,})['\"]")
_FINAL_STATE_KEYWORDS = (
    "present",
    "not present",
    "absent",
    "saved",
    "sent",
    "stored",
    "visible",
    "shown",
    "appears",
    "exists",
    "created",
    "added",
    "moved",
    "deleted",
    "removed",
)
_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "within",
    "same",
    "this",
    "that",
    "these",
    "those",
    "task",
    "entry",
    "expense",
    "field",
    "fields",
    "value",
    "values",
    "text",
    "display",
    "displays",
    "shows",
    "show",
    "contains",
    "contain",
    "selection",
    "set",
    "indicates",
    "system",
    "directory",
    "directories",
    "folder",
    "folders",
    "file",
    "files",
    "storage",
    "area",
    "source",
    "target",
    "named",
    "requested",
    "single",
    "name",
    "present",
    "listed",
    "execution",
    "trace",
    "under",
    "positive",
    "negative",
    "evidence",
    "step",
    "steps",
    "observable",
    "requirement",
    "requirements",
}
_NEGATIVE_EVIDENCE_CUES = (
    "no match",
    "no matches",
    "not present",
    "not found",
    "cannot find",
    "can't find",
    "missing",
    "infeasible",
    "not visible",
)
_POSITIVE_EVIDENCE_CUES = (
    "moved",
    "saved",
    "sent",
    "stored",
    "created",
    "added",
    "completed",
    "visible",
    "shown",
    "present",
)


def serialize_step_data(
    step_data: dict[str, Any] | None,
    step_number: int,
    *,
    max_ui_elements: int = DEFAULT_MAX_UI_ELEMENTS,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    screenshot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Serializes one T3A/M3A AgentInteractionResult.data payload."""
    step_data = step_data or {}
    action_output = _trim_text(step_data.get("action_output"), max_text_chars)
    reason, parsed_action = _parse_reason_action(action_output)
    action = _action_to_dict(step_data.get("action_output_json")) or parsed_action

    if not reason:
        reason = _trim_text(step_data.get("action_reason"), max_text_chars)

    before_screenshot_path = _save_screenshot(
        step_data.get("before_screenshot"),
        screenshot_dir=screenshot_dir,
        step_number=step_number,
        kind="before",
    )
    after_screenshot_path = _save_screenshot(
        step_data.get("after_screenshot"),
        screenshot_dir=screenshot_dir,
        step_number=step_number,
        kind="after",
    )

    return {
        "step": step_number,
        "action": action,
        "reason": reason,
        "action_output": action_output,
        "summary": _trim_text(step_data.get("summary"), max_text_chars),
        "before_ui": _serialize_ui_elements(
            step_data.get("before_element_list")
            or step_data.get("before_ui_elements"),
            max_items=max_ui_elements,
        ),
        "after_ui": _serialize_ui_elements(
            step_data.get("after_element_list")
            or step_data.get("after_ui_elements"),
            max_items=max_ui_elements,
        ),
        "before_screenshot_path": before_screenshot_path,
        "after_screenshot_path": after_screenshot_path,
    }


def serialize_error_step(step_number: int, error: Exception | str) -> dict[str, Any]:
    """Creates a trace step for runner-level exceptions."""
    return {
        "step": step_number,
        "action": None,
        "reason": "",
        "action_output": "",
        "summary": "Runner caught an exception before this step completed.",
        "before_ui": [],
        "after_ui": [],
        "error": str(error),
    }


def make_trace(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds the trace envelope stored on each result record."""
    return {
        "format_version": TRACE_FORMAT_VERSION,
        "steps": steps,
    }


def compact_trial_for_prompt(record: dict[str, Any], *, max_steps: int = 60) -> dict[str, Any]:
    """Returns the compact trial payload sent to the judge model."""
    payload, _ = build_trial_payload_and_images(record, max_steps=max_steps)
    return payload


def build_trial_payload_and_images(
    record: dict[str, Any],
    *,
    max_steps: int = 60,
    max_images: int = DEFAULT_MAX_PROMPT_IMAGES,
) -> tuple[dict[str, Any], list[str]]:
    """Returns the compact trial payload and selected screenshot paths."""
    trace = record.get("trace") or {}
    steps = trace.get("steps") or []
    clipped_steps = steps[:max_steps]
    prompt_images, manifest, screenshot_truncated = _collect_prompt_images(
        clipped_steps,
        max_images=max_images,
    )
    return {
        "task": record.get("task"),
        "granularity": record.get("granularity"),
        "goal_used": _trim_text(record.get("goal_used"), DEFAULT_MAX_TEXT_CHARS),
        "official_reward": record.get("reward"),
        "official_success": record.get("success"),
        "agent_done": record.get("agent_done"),
        "abort_reason": record.get("abort_reason"),
        "steps_taken": record.get("steps"),
        "trace_format_version": trace.get("format_version"),
        "trace_steps": [_step_for_prompt(step) for step in clipped_steps],
        "trace_truncated": len(steps) > max_steps,
        "trace_screenshots_attached": len(prompt_images),
        "trace_screenshots_truncated": screenshot_truncated,
        "trace_screenshot_manifest": manifest,
    }, prompt_images


def build_checkpoint_payload_and_images(
    record: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    max_steps: int = DEFAULT_MAX_CHECKPOINT_EVIDENCE_STEPS,
    max_images: int = DEFAULT_MAX_CHECKPOINT_IMAGES,
    neighbor_window: int = DEFAULT_EVIDENCE_NEIGHBOR_WINDOW,
) -> tuple[dict[str, Any], list[str]]:
    """Returns checkpoint-specific evidence and selected screenshot paths."""
    trace = record.get("trace") or {}
    steps = trace.get("steps") or []
    query_tokens = _checkpoint_query_tokens(checkpoint)
    query_phrases = _checkpoint_query_phrases(checkpoint)
    selected_indices = _select_checkpoint_evidence_indices(
        steps,
        checkpoint,
        limit=max_steps,
        neighbor_window=neighbor_window,
    )
    selected_steps = [steps[index] for index in selected_indices]
    prompt_images, manifest, screenshot_truncated = _collect_prompt_images(
        selected_steps,
        max_images=max_images,
    )
    return {
        "task": record.get("task"),
        "granularity": record.get("granularity"),
        "goal_used": _trim_text(record.get("goal_used"), DEFAULT_MAX_TEXT_CHARS),
        "official_reward": record.get("reward"),
        "official_success": record.get("success"),
        "steps_taken": record.get("steps"),
        "trace_format_version": trace.get("format_version"),
        "trace_total_steps": len(steps),
        "selected_step_numbers": [step.get("step") for step in selected_steps],
        "selection_strategy": "checkpoint_relevance_v2",
        "checkpoint_focus": {
            "id": checkpoint.get("id"),
            "description": checkpoint.get("description"),
            "evidence_hint": checkpoint.get("evidence_hint"),
            "required": checkpoint.get("required", True),
        },
        "trace_steps": [
            _checkpoint_step_for_prompt(
                step,
                query_tokens=query_tokens,
                query_phrases=query_phrases,
            )
            for step in selected_steps
        ],
        "trace_truncated": len(steps) > len(selected_steps),
        "trace_screenshots_attached": len(prompt_images),
        "trace_screenshots_truncated": screenshot_truncated,
        "trace_screenshot_manifest": manifest,
    }, prompt_images


def _parse_reason_action(action_output: str) -> tuple[str, dict[str, Any] | None]:
    if not action_output:
        return "", None
    match = _REASON_ACTION_RE.search(action_output)
    if not match:
        return "", _extract_first_json_object(action_output)
    return match.group(1).strip(), _extract_first_json_object(match.group(2).strip())


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    first_brace = text.find("{")
    if first_brace < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[first_brace:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _step_for_prompt(step: dict[str, Any]) -> dict[str, Any]:
    filtered = {
        key: value
        for key, value in step.items()
        if key not in {"before_screenshot_path", "after_screenshot_path"}
    }
    filtered["has_before_screenshot"] = bool(step.get("before_screenshot_path"))
    filtered["has_after_screenshot"] = bool(step.get("after_screenshot_path"))
    return filtered


def _checkpoint_step_for_prompt(
    step: dict[str, Any],
    *,
    query_tokens: set[str],
    query_phrases: list[str],
) -> dict[str, Any]:
    return {
        "step": step.get("step"),
        "action": step.get("action"),
        "reason": _checkpoint_reason_for_prompt(
            step,
            query_tokens=query_tokens,
            query_phrases=query_phrases,
        ),
        "summary": _trim_text(
            step.get("summary"),
            DEFAULT_MAX_CHECKPOINT_SUMMARY_CHARS,
        ),
        "before_ui": _checkpoint_ui_for_prompt(
            step.get("before_ui") or [],
            query_tokens=query_tokens,
            query_phrases=query_phrases,
        ),
        "after_ui": _checkpoint_ui_for_prompt(
            step.get("after_ui") or [],
            query_tokens=query_tokens,
            query_phrases=query_phrases,
        ),
        "has_before_screenshot": bool(step.get("before_screenshot_path")),
        "has_after_screenshot": bool(step.get("after_screenshot_path")),
        **({"error": str(step.get("error"))} if step.get("error") else {}),
    }


def _action_to_dict(action: Any) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, dict):
        return _json_safe_dict(action)
    if hasattr(action, "as_dict"):
        try:
            return _json_safe_dict(action.as_dict(skip_none=True))
        except TypeError:
            return _json_safe_dict(action.as_dict())
    if dataclasses.is_dataclass(action):
        return _json_safe_dict(dataclasses.asdict(action))
    return None


def _select_checkpoint_evidence_indices(
    steps: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    *,
    limit: int,
    neighbor_window: int,
) -> list[int]:
    if not steps or limit <= 0:
        return []

    query_tokens = _checkpoint_query_tokens(checkpoint)
    query_phrases = _checkpoint_query_phrases(checkpoint)
    profile = _checkpoint_profile(checkpoint)
    selected: list[int] = []

    for reserved in _reserved_tail_indices(
        steps,
        query_tokens,
        query_phrases,
        profile,
        limit=min(limit, DEFAULT_FINAL_STATE_TAIL_STEPS),
    ):
        if reserved not in selected:
            selected.append(reserved)
        if len(selected) >= limit:
            return sorted(selected[:limit])

    for anchor in _precision_anchor_indices(
        steps,
        query_tokens=query_tokens,
        query_phrases=query_phrases,
        limit=min(3, limit),
        neighbor_window=neighbor_window,
    ):
        if anchor not in selected:
            selected.append(anchor)
        if len(selected) >= limit:
            return sorted(selected[:limit])

    scored_indices = sorted(
        (
            (
                index,
                _step_relevance_score(
                    step,
                    query_tokens,
                    query_phrases,
                    profile=profile,
                    step_index=index,
                    total_steps=len(steps),
                ),
            )
            for index, step in enumerate(steps)
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )

    for index, score in scored_indices:
        if score <= 0:
            break
        if _overlaps_selected_region(index, selected, neighbor_window=neighbor_window):
            continue
        for neighbor in range(index - neighbor_window, index + neighbor_window + 1):
            if 0 <= neighbor < len(steps) and neighbor not in selected:
                selected.append(neighbor)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    for fallback in _fallback_evidence_indices(steps, profile=profile):
        if fallback not in selected:
            selected.append(fallback)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for index in _select_evenly_spaced_indices(list(range(len(steps))), limit=limit):
            if index not in selected:
                selected.append(index)
            if len(selected) >= limit:
                break

    return sorted(selected[:limit])


def _collect_prompt_images(
    steps: list[dict[str, Any]],
    *,
    max_images: int,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    candidate_indices = [
        index
        for index, step in enumerate(steps)
        if _evidence_screenshot_path(step)
    ]
    if not candidate_indices:
        return [], [], False

    selected_indices = _select_evenly_spaced_indices(
        candidate_indices,
        limit=max_images,
    )
    prompt_images: list[str] = []
    manifest: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for step_index in selected_indices:
        step = steps[step_index]
        path = _evidence_screenshot_path(step)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        image_index = len(prompt_images) + 1
        prompt_images.append(path)
        manifest.append(
            {
                "image_index": image_index,
                "step": step.get("step"),
                "kind": "after" if step.get("after_screenshot_path") else "before",
            }
        )
    return prompt_images, manifest, len(candidate_indices) > len(prompt_images)


def _evidence_screenshot_path(step: dict[str, Any]) -> str:
    raw = str(step.get("after_screenshot_path") or step.get("before_screenshot_path") or "")
    if not raw:
        return ""
    return raw if Path(raw).exists() else ""


def _checkpoint_query_tokens(checkpoint: dict[str, Any]) -> set[str]:
    text = " ".join(
        str(checkpoint.get(key) or "")
        for key in ("description", "evidence_hint", "id")
    )
    tokens = set()
    for raw_token in _tokenize(text):
        token = _normalize_token_for_match(raw_token)
        if not token:
            continue
        if not (len(token) >= 3 or token.isdigit() or any(char.isdigit() for char in token)):
            continue
        if token in _TOKEN_STOPWORDS or re.fullmatch(r"cp\d+", token):
            continue
        tokens.add(token)
    return tokens


def _checkpoint_query_phrases(checkpoint: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(checkpoint.get(key) or "")
        for key in ("description", "evidence_hint")
    )
    phrases = []
    for match in _QUOTED_TEXT_RE.finditer(text):
        phrase = match.group(1).strip().lower()
        if len(phrase) >= 3:
            phrases.append(phrase)
    return phrases


def _checkpoint_profile(checkpoint: dict[str, Any]) -> dict[str, bool]:
    text = " ".join(
        str(checkpoint.get(key) or "").lower()
        for key in ("description", "evidence_hint")
    )
    absence = any(keyword in text for keyword in ("not present", "absent", "removed", "deleted"))
    final_state = absence or any(keyword in text for keyword in _FINAL_STATE_KEYWORDS)
    return {
        "absence": absence,
        "final_state": final_state,
    }


def _step_relevance_score(
    step: dict[str, Any],
    query_tokens: set[str],
    query_phrases: list[str],
    *,
    profile: dict[str, bool],
    step_index: int,
    total_steps: int,
) -> float:
    if not query_tokens:
        return 0.0

    sources = _step_search_sources(step)
    haystack = " ".join(sources.values())
    score = 0.0
    score += _query_match_score(
        sources["action"] + " " + sources["summary"],
        query_tokens=query_tokens,
        query_phrases=query_phrases,
        source_weight=1.0,
    )
    score += _query_match_score(
        sources["ui"],
        query_tokens=query_tokens,
        query_phrases=query_phrases,
        source_weight=1.5,
    )
    score += _query_match_score(
        sources["reason"],
        query_tokens=query_tokens,
        query_phrases=query_phrases,
        source_weight=0.2,
    )
    action = step.get("action") or {}
    action_type = str(action.get("action_type") or "").lower()
    if action_type in {"status", "answer"}:
        score += 0.75 if profile["final_state"] else 0.5
    elif action_type in {"click", "input_text", "long_press"}:
        score += 0.3
    if step.get("after_ui"):
        score += 0.15
    if step.get("after_screenshot_path") or step.get("before_screenshot_path"):
        score += 0.1
    if profile["final_state"] and total_steps > 1:
        score += 1.5 * (step_index / (total_steps - 1))
    if _has_negative_evidence_cue(haystack):
        score += 2.0 if profile["absence"] else 1.1
    if profile["final_state"] and _has_positive_evidence_cue(haystack):
        score += 0.6
    if "search" in haystack and profile["final_state"]:
        score += 0.4
    return score


def _step_search_text(step: dict[str, Any]) -> str:
    sources = _step_search_sources(step)
    return " ".join(sources.values())


def _step_search_sources(step: dict[str, Any]) -> dict[str, str]:
    return {
        "action": json.dumps(step.get("action") or {}, ensure_ascii=False, sort_keys=True).lower(),
        "reason": str(step.get("reason") or "").lower(),
        "summary": str(step.get("summary") or "").lower(),
        "ui": " ".join(
            str(item)
            for item in (step.get("before_ui") or []) + (step.get("after_ui") or [])
        ).lower(),
    }


def _checkpoint_reason_for_prompt(
    step: dict[str, Any],
    *,
    query_tokens: set[str],
    query_phrases: list[str],
) -> str:
    reason = str(step.get("reason") or "")
    if not reason:
        return ""
    lowered = reason.lower()
    if _has_negative_evidence_cue(lowered) or _has_positive_evidence_cue(lowered):
        return _trim_text(reason, DEFAULT_MAX_CHECKPOINT_REASON_CHARS)
    if any(phrase in lowered for phrase in query_phrases):
        return _trim_text(reason, DEFAULT_MAX_CHECKPOINT_REASON_CHARS)
    if any(token in lowered for token in query_tokens):
        return _trim_text(reason, DEFAULT_MAX_CHECKPOINT_REASON_CHARS)
    return ""


def _checkpoint_ui_for_prompt(
    elements: list[str],
    *,
    query_tokens: set[str],
    query_phrases: list[str],
) -> list[str]:
    matched: list[str] = []
    fallback: list[str] = []
    for element in elements:
        text = str(element)
        lowered = text.lower()
        trimmed = _trim_text(text, DEFAULT_MAX_CHECKPOINT_UI_CHARS)
        if any(phrase in lowered for phrase in query_phrases) or any(
            token in lowered for token in query_tokens
        ):
            matched.append(trimmed)
        elif len(fallback) < DEFAULT_MAX_CHECKPOINT_UI_FALLBACK:
            fallback.append(trimmed)
    if matched:
        return matched[:DEFAULT_MAX_CHECKPOINT_UI_ELEMENTS]
    return fallback


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


def _query_token_weight(token: str, *, exact: bool) -> float:
    if exact:
        if _is_entity_like_token(token):
            return 3.0
        if len(token) >= 8:
            return 1.5
        return 1.0
    if _is_entity_like_token(token):
        return 0.75
    return 0.35


def _query_match_score(
    text: str,
    *,
    query_tokens: set[str],
    query_phrases: list[str],
    source_weight: float,
) -> float:
    if not text:
        return 0.0
    text_tokens = {
        normalized
        for raw_token in _tokenize(text)
        if (normalized := _normalize_token_for_match(raw_token))
    }
    phrase_matches = [phrase for phrase in query_phrases if phrase in text]
    exact_matches = {
        token for token in query_tokens if _token_variants(token) & text_tokens
    }
    partial_matches = [
        token for token in query_tokens if token not in exact_matches and token in text
    ]
    score = sum(_query_token_weight(token, exact=True) for token in exact_matches)
    score += sum(_query_token_weight(token, exact=False) for token in partial_matches)
    score += 4.0 * len(phrase_matches)
    return source_weight * score


def _is_entity_like_token(token: str) -> bool:
    return any(char.isdigit() for char in token) or any(char in token for char in "._/-+")


def _normalize_token_for_match(token: str) -> str:
    return str(token or "").lower().strip(".,;:!?()[]{}")


def _token_variants(token: str) -> set[str]:
    normalized = _normalize_token_for_match(token)
    if not normalized:
        return set()
    variants = {normalized}
    if (
        normalized.endswith("s")
        and len(normalized) > 4
        and not normalized.endswith("ss")
    ):
        variants.add(normalized[:-1])
    elif len(normalized) > 4:
        variants.add(normalized + "s")
    return variants


def _is_contextual_query_token(token: str) -> bool:
    normalized = _normalize_token_for_match(token)
    return normalized.startswith("sdk_") or "arm64" in normalized


def _has_negative_evidence_cue(haystack: str) -> bool:
    return any(cue in haystack for cue in _NEGATIVE_EVIDENCE_CUES)


def _has_positive_evidence_cue(haystack: str) -> bool:
    return any(cue in haystack for cue in _POSITIVE_EVIDENCE_CUES)


def _reserved_tail_indices(
    steps: list[dict[str, Any]],
    query_tokens: set[str],
    query_phrases: list[str],
    profile: dict[str, bool],
    *,
    limit: int,
) -> list[int]:
    if not steps or limit <= 0 or not profile["final_state"]:
        return []
    tail_start = max(0, len(steps) - DEFAULT_FINAL_STATE_TAIL_STEPS)
    tail_indices = list(range(tail_start, len(steps)))
    exact_tail = [
        index
        for index in tail_indices
        if (
            query_tokens & set(_tokenize(_step_search_text(steps[index])))
            or any(phrase in _step_search_text(steps[index]) for phrase in query_phrases)
        )
    ]
    search_tail = [
        index
        for index in tail_indices
        if "search" in _step_search_text(steps[index])
        or _has_negative_evidence_cue(_step_search_text(steps[index]))
    ]
    status_tail = [
        index
        for index in tail_indices
        if str((steps[index].get("action") or {}).get("action_type") or "").lower()
        in {"status", "answer"}
    ]
    ordered = exact_tail + search_tail + status_tail + tail_indices
    reserved: list[int] = []
    for index in ordered:
        if index not in reserved:
            reserved.append(index)
        if len(reserved) >= limit:
            break
    return reserved


def _precision_anchor_indices(
    steps: list[dict[str, Any]],
    *,
    query_tokens: set[str],
    query_phrases: list[str],
    limit: int,
    neighbor_window: int,
) -> list[int]:
    if not steps or limit <= 0:
        return []
    scored: list[tuple[float, int]] = []
    signal_tokens = {token for token in query_tokens if not _is_contextual_query_token(token)}
    for index, step in enumerate(steps):
        sources = _step_search_sources(step)
        core_text = " ".join((sources["action"], sources["summary"], sources["ui"]))
        core_tokens = {
            normalized
            for raw_token in _tokenize(core_text)
            if (normalized := _normalize_token_for_match(raw_token))
        }
        token_matches = {
            token
            for token in (signal_tokens or query_tokens)
            if _token_variants(token) & core_tokens or token in core_text
        }
        phrase_matches = [phrase for phrase in query_phrases if phrase in core_text]
        if not phrase_matches and len(token_matches) < 2:
            continue
        action_type = str((step.get("action") or {}).get("action_type") or "").lower()
        score = (5.0 * len(phrase_matches)) + (3.0 * len(token_matches))
        if action_type in {"click", "input_text", "long_press", "status"}:
            score += 1.5
        if _has_positive_evidence_cue(core_text):
            score += 2.0
        scored.append((score + (index / max(1, len(steps))), index))

    selected: list[int] = []
    for _, index in sorted(scored, reverse=True):
        if _overlaps_selected_region(index, selected, neighbor_window=neighbor_window):
            continue
        selected.append(index)
        if len(selected) >= limit:
            break
    return selected


def _fallback_evidence_indices(
    steps: list[dict[str, Any]],
    *,
    profile: dict[str, bool],
) -> list[int]:
    if not steps:
        return []
    tail = list(range(max(0, len(steps) - 3), len(steps)))
    status_steps = [
        index
        for index, step in enumerate(steps)
        if str((step.get("action") or {}).get("action_type") or "").lower() == "status"
    ]
    screenshot_steps = [
        index
        for index, step in enumerate(steps)
        if _evidence_screenshot_path(step)
    ]
    ordered = tail + status_steps[::-1] + screenshot_steps[::-1]
    if profile["final_state"]:
        ordered = tail[::-1] + ordered
    deduped: list[int] = []
    for index in ordered:
        if index not in deduped:
            deduped.append(index)
    return deduped


def _select_evenly_spaced_indices(indices: list[int], *, limit: int) -> list[int]:
    if limit <= 0:
        return []
    if len(indices) <= limit:
        return indices
    if limit == 1:
        return [indices[-1]]
    picks = []
    last_pos = -1
    for slot in range(limit):
        pos = round(slot * (len(indices) - 1) / (limit - 1))
        if pos <= last_pos:
            pos = last_pos + 1
        pos = min(pos, len(indices) - 1)
        picks.append(indices[pos])
        last_pos = pos
    return picks


def _overlaps_selected_region(
    index: int,
    selected: list[int],
    *,
    neighbor_window: int,
) -> bool:
    if not selected:
        return False
    buffer = max(1, (2 * neighbor_window) + 1)
    return any(abs(index - existing) <= buffer for existing in selected)


def _serialize_ui_elements(elements: Any, *, max_items: int) -> list[str]:
    if not elements:
        return []
    if isinstance(elements, str):
        lines = [line.strip() for line in elements.splitlines() if line.strip()]
        return [_trim_text(line, DEFAULT_MAX_ELEMENT_CHARS) for line in lines[:max_items]]
    if not isinstance(elements, (list, tuple)):
        return [_trim_text(elements, DEFAULT_MAX_ELEMENT_CHARS)]
    return [
        _trim_text(element, DEFAULT_MAX_ELEMENT_CHARS)
        for element in elements[:max_items]
    ]


def _save_screenshot(
    screenshot: Any,
    *,
    screenshot_dir: str | Path | None,
    step_number: int,
    kind: str,
) -> str:
    if screenshot_dir is None or screenshot is None:
        return ""
    path = Path(screenshot_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"step_{step_number:03d}_{kind}.jpg"
    image = Image.fromarray(screenshot)
    image.save(output_path, format="JPEG", quality=85)
    return str(output_path)


def _json_safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(subvalue)
        for key, subvalue in value.items()
        if subvalue is not None
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return _json_safe_dict(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    return str(value)


def _trim_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...[truncated]"
