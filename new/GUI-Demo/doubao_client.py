"""Doubao Ark SDK client used by the AndroidWorld GUI runner."""

from __future__ import annotations

import json
import os
import time
from typing import Any


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_EXECUTION_MODEL = "doubao-seed-1-6-vision-250815"


def _count_image_parts(messages: list[dict[str, Any]]) -> int:
    """Count image_url content parts in a chat-completions message list."""
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                count += 1
    return count


def _extract_usage_tokens(response: Any) -> tuple[int | None, int | None, int | None]:
    """Best-effort prompt/completion/total token extraction."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    if isinstance(usage, dict):
        return (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )


def _message_content_to_str(content: Any) -> str:
    """Normalize Ark/OpenAI message.content (str or list of content parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _field(obj: Any, name: str) -> Any:
    """Read one field from either an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _reasoning_to_str(value: Any) -> str:
    """Normalize possible Ark/OpenAI reasoning/thinking payload shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_reasoning_to_str(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "reasoning_content",
            "reasoning",
            "thinking",
            "summary",
        ):
            text = _reasoning_to_str(value.get(key))
            if text:
                return text
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value).strip()


def _message_to_text_parts(message: Any) -> tuple[str, str]:
    """Return (content, reasoning) from SDK message objects.

    Doubao/Ark may put the visible answer in `content` and model thinking in
    provider-specific fields such as `reasoning_content`. Some SDK versions
    expose those as object attributes, while others expose dict-like payloads.
    """
    content = _message_content_to_str(_field(message, "content")).strip()
    reasoning_fields = (
        "reasoning_content",
        "reasoning",
        "thinking",
        "reasoning_details",
    )
    for name in reasoning_fields:
        reasoning = _reasoning_to_str(_field(message, name))
        if reasoning:
            return content, reasoning

    dump = None
    for dump_method in ("model_dump", "to_dict"):
        method = getattr(message, dump_method, None)
        if callable(method):
            try:
                dump = method()
                break
            except Exception:  # pylint: disable=broad-exception-caught
                pass
    if isinstance(dump, dict):
        for name in reasoning_fields:
            reasoning = _reasoning_to_str(dump.get(name))
            if reasoning:
                return content, reasoning
    return content, ""


def _compose_gui_completion(content: str, reasoning: str) -> str:
    """Prefer visible content, but preserve provider reasoning for trace thought."""
    content = content.strip()
    reasoning = reasoning.strip()
    if content:
        has_thought = "thought" in content.lower()
        if reasoning and not has_thought:
            return f"Thought: {reasoning}\n{content}"
        return content
    return reasoning


class DoubaoClient:
    """Small wrapper around the official Volcengine Ark SDK."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EXECUTION_MODEL,
        api_key: str | None = None,
        api_key_env: str = "ARK_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        timeout_seconds: int = 120,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as exc:
            raise RuntimeError(
                "Ark SDK is not installed. Run: pip install 'volcengine-python-sdk[ark]'"
            ) from exc

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"No API key provided and {api_key_env} is not set.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.extra_body = extra_body
        self.client = Ark(api_key=key, base_url=base_url, timeout=timeout_seconds)

    def complete(self, messages: list[dict[str, Any]]) -> str:
        """Return one non-empty model completion."""
        last_error: Exception | None = None
        image_count = _count_image_parts(messages)
        for attempt in range(1, self.max_retries + 1):
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.extra_body:
                    create_kwargs["extra_body"] = self.extra_body
                started = time.perf_counter()
                response = self.client.chat.completions.create(**create_kwargs)
                latency_s = time.perf_counter() - started
                prompt_tokens, completion_tokens, total_tokens = _extract_usage_tokens(response)
                print(
                    "[LLM_METRICS] "
                    f"model={self.model} "
                    f"latency_s={latency_s:.3f} "
                    f"prompt_tokens={prompt_tokens} "
                    f"completion_tokens={completion_tokens} "
                    f"total_tokens={total_tokens} "
                    f"images={image_count} "
                    f"attempt={attempt}"
                )
                content, reasoning = _message_to_text_parts(response.choices[0].message)
                text = _compose_gui_completion(content, reasoning)
                if text.strip():
                    return text
                raise RuntimeError("Model returned empty or unparseable content.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        if last_error is not None:
            raise last_error
        raise RuntimeError("DoubaoClient.complete failed with no last_error.")
