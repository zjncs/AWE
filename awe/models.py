"""Text model adapters used by the trace evaluator."""

from __future__ import annotations

import base64
from pathlib import Path
import time
from typing import Any, Protocol


class TextModel(Protocol):
    """Minimal interface for evaluator LLM calls."""

    def complete(self, prompt: str, images: list[str] | None = None) -> str:
        """Returns model text for one prompt."""


class OpenAIChatModel:
    """OpenAI-compatible chat/completions client with retry."""

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
        initial_retry_delay: float = 2.0,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.client = OpenAI(
            api_key=api_key,
            base_url=_normalize_base_url(base_url),
            timeout=timeout_seconds,
        )

    def complete(self, prompt: str, images: list[str] | None = None) -> str:
        last_error: Exception | None = None
        delay = self.initial_retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for image_path in images or []:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_path_to_data_url(image_path),
                            },
                        }
                    )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": content}],
                    extra_body=self.extra_body or None,
                )
                if not response.choices:
                    raise RuntimeError("Model returned no choices.")
                message = response.choices[0].message
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content
                raise RuntimeError(
                    f"Empty content in response (attempt {attempt})"
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    print(f"  [judge retry {attempt}/{self.max_retries}] {exc}")
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
        raise last_error  # type: ignore[misc]


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def _image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
