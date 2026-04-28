"""Chat model adapters."""

from __future__ import annotations

import time
from typing import Any, Protocol


class ChatModel(Protocol):
    """Minimal message-based model interface."""

    def complete(self, messages: list[dict[str, Any]]) -> str:
        """Return model text for one message list."""


def _message_content_to_str(content: Any) -> str:
    """Normalize SDK message.content into plain text."""
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
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def _field(obj: Any, name: str) -> Any:
    """Read one field from either an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _reasoning_to_str(value: Any) -> str:
    """Normalize possible Ark/OpenAI reasoning payload shapes."""
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
        return str(value)
    return str(value).strip()


def _message_to_text(message: Any) -> str:
    """Extract visible content, falling back to provider reasoning fields.

    Evaluation prompts expect parseable JSON. If `content` exists, we keep it
    unchanged. Reasoning is used only when the provider returned an empty
    content field, which avoids appending non-JSON prose to valid JSON.
    """
    content = _message_content_to_str(_field(message, "content")).strip()
    if content:
        return content

    reasoning_fields = (
        "reasoning_content",
        "reasoning",
        "thinking",
        "reasoning_details",
    )
    for name in reasoning_fields:
        reasoning = _reasoning_to_str(_field(message, name))
        if reasoning:
            return reasoning

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
                return reasoning
    return ""


class ArkChatModel:
    """Official Volcengine Ark SDK adapter."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout_seconds: int = 180,
        extra_body: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as exc:
            raise RuntimeError(
                "Ark SDK is not installed. Run: pip install 'volcengine-python-sdk[ark]'"
            ) from exc

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.client = Ark(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    def complete(self, messages: list[dict[str, Any]]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.extra_body:
                    create_kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**create_kwargs)
                content = _message_to_text(response.choices[0].message)
                if content.strip():
                    return content
                raise RuntimeError("Model returned empty content.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 20))
        raise last_error  # type: ignore[misc]


class OpenAICompatibleChatModel:
    """OpenAI-compatible adapter for Ark's /api/v3 endpoint."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout_seconds: int = 180,
        extra_body: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    def complete(self, messages: list[dict[str, Any]]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=self.extra_body or None,
                )
                content = _message_to_text(response.choices[0].message)
                if content.strip():
                    return content
                raise RuntimeError("Model returned empty content.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 20))
        raise last_error  # type: ignore[misc]
