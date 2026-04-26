"""Chat model adapters."""

from __future__ import annotations

import time
from typing import Any, Protocol


class ChatModel(Protocol):
    """Minimal message-based model interface."""

    def complete(self, messages: list[dict[str, Any]]) -> str:
        """Return model text for one message list."""


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
        self.max_retries = max_retries
        self.client = Ark(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    def complete(self, messages: list[dict[str, Any]]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
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
                content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content
                raise RuntimeError("Model returned empty content.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 20))
        raise last_error  # type: ignore[misc]
