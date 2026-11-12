"""Shared helpers for HTTP providers."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import ClassVar

import requests

from .base import RequestSpec


def auth_headers(spec: RequestSpec) -> dict:
    h = {"Content-Type": "application/json"}
    if spec.api_key:
        h["Authorization"] = f"Bearer {spec.api_key}"
    return h


def stream_sse(
    iter_lines: Iterator[bytes], content_extractor, first_token_only: bool = False
) -> tuple[str, float, dict | None, bool]:
    """Consume an SSE stream.

    content_extractor(payload) -> (text_delta, usage_dict_or_None, done_bool)
    Returns (full_text, ttft_seconds, usage, completed).
    With first_token_only=True, stops after the first content chunk
    (completed=False) — used for TTFT-focused scenarios.
    """
    text_parts: list[str] = []
    ttft: float | None = None
    usage = None
    completed = False
    start = time.monotonic()
    for raw in iter_lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line or line.startswith(":"):
            continue
        if line == "data: [DONE]":
            completed = True
            break
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        try:
            delta_text, usage_chunk, done = content_extractor(payload)
        except Exception:
            continue
        if usage_chunk:
            if usage is None:
                usage = dict(usage_chunk)
            else:
                usage.update({k: v for k, v in usage_chunk.items() if v is not None})
        if delta_text:
            if ttft is None:
                ttft = time.monotonic() - start
            text_parts.append(delta_text)
            if first_token_only:
                return "".join(text_parts), ttft, usage, False
        if done:
            completed = True
            break
    if ttft is None:
        ttft = time.monotonic() - start
    return "".join(text_parts), ttft, usage, completed


def openai_sse_extractor(payload: str):
    """Extract (text_delta, usage, done) from an OpenAI chat.completion.chunk."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return "", None, False
    usage = obj.get("usage")
    choices = obj.get("choices") or []
    if not choices:
        # usage-only final chunk (stream_options.include_usage)
        return "", usage, bool(usage)
    delta = (choices[0].get("delta") or {}).get("content") or ""
    finish = choices[0].get("finish_reason")
    return delta, usage, bool(finish)


def anthropic_sse_extractor(payload: str):
    """Anthropic streams events as `event: X\\ndata: {...}` — we only see data lines here;
    the SSE parser must pass the event type too. We handle both by sniffing JSON type."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return "", None, False
    t = obj.get("type")
    if t == "content_block_delta":
        delta = ((obj.get("delta") or {}).get("text")) or ""
        return delta, None, False
    if t == "message_delta":
        usage = obj.get("usage")
        return "", usage, bool(obj.get("delta", {}).get("stop_reason"))
    if t == "message_start":
        usage = (obj.get("message") or {}).get("usage")
        return "", usage, False
    if t == "message_stop":
        return "", None, True
    return "", None, False


class HttpClient:
    """Thin requests wrapper; sessions are shared per timeout for conn reuse."""

    _sessions: ClassVar[dict[float, requests.Session]] = {}

    def __init__(self, timeout_s: float = 120.0):
        self.timeout = timeout_s
        if timeout_s not in self._sessions:
            self._sessions[timeout_s] = requests.Session()
        self.session = self._sessions[timeout_s]

    def post(
        self,
        url: str,
        headers: dict,
        body: dict | bytes | None,
        stream: bool = False,
        timeout: float | None = None,
    ) -> requests.Response:
        return self.session.post(
            url,
            headers=headers,
            json=body if isinstance(body, dict) else None,
            data=body if isinstance(body, (bytes, str)) else None,
            stream=stream,
            timeout=timeout or self.timeout,
        )
