"""Shared helpers for HTTP providers."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar

import requests

from .base import RequestSpec


@dataclass
class StreamResult:
    """Result of consuming an SSE stream, including per-chunk timing.

    chunk_offsets_s holds the offset (seconds from stream start) of every
    content-bearing chunk — the raw material for TPOT / inter-token latency /
    generation-duration metrics.
    """

    text: str = ""
    ttft_s: float | None = None
    usage: dict | None = None
    completed: bool = False
    e2e_s: float = 0.0
    chunk_offsets_s: list[float] = field(default_factory=list)


def auth_headers(spec: RequestSpec) -> dict:
    h = {"Content-Type": "application/json"}
    if spec.api_key:
        h["Authorization"] = f"Bearer {spec.api_key}"
    return h


def stream_sse(
    iter_lines: Iterator[bytes], content_extractor, first_token_only: bool = False
) -> StreamResult:
    """Consume an SSE stream.

    content_extractor(payload) -> (text_delta, usage_dict_or_None, done_bool)
    Returns a StreamResult with full text, TTFT, merged usage, completion flag,
    total elapsed time, and per-chunk offsets for TPOT/ITL derivation.
    With first_token_only=True, stops after the first content chunk
    (completed=False) — used for TTFT-focused scenarios.
    """
    text_parts: list[str] = []
    ttft: float | None = None
    usage = None
    completed = False
    chunk_offsets: list[float] = []
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
            now = time.monotonic() - start
            if ttft is None:
                ttft = now
            text_parts.append(delta_text)
            chunk_offsets.append(now)
            if first_token_only:
                return StreamResult(
                    text="".join(text_parts),
                    ttft_s=ttft,
                    usage=usage,
                    completed=False,
                    e2e_s=time.monotonic() - start,
                    chunk_offsets_s=chunk_offsets,
                )
        if done:
            completed = True
            break
    e2e = time.monotonic() - start
    if ttft is None:
        ttft = e2e
    return StreamResult(
        text="".join(text_parts),
        ttft_s=ttft,
        usage=usage,
        completed=completed,
        e2e_s=e2e,
        chunk_offsets_s=chunk_offsets,
    )


def stream_timing_stats(chunk_offsets_s: list[float], completion_tokens: int) -> dict:
    """Derive TPOT / inter-token-latency / generation-duration stats.

    Returns dict with generation_s, tpot_s, itl_avg_ms, itl_p50_ms, itl_p95_ms,
    itl_max_ms (all None-safe). ITL is measured per SSE content chunk — a chunk
    may carry multiple tokens, so this is chunk cadence, not strict per-token.
    """
    out = {
        "generation_s": None,
        "tpot_s": None,
        "itl_avg_ms": None,
        "itl_p50_ms": None,
        "itl_p95_ms": None,
        "itl_max_ms": None,
        "chunks": len(chunk_offsets_s),
    }
    offs = chunk_offsets_s
    if len(offs) < 2:
        return out
    generation_s = offs[-1] - offs[0]
    out["generation_s"] = generation_s
    itls = [b - a for a, b in pairwise(offs)]
    itls_ms = sorted(i * 1000.0 for i in itls)
    n = len(itls_ms)
    out["itl_avg_ms"] = sum(itls_ms) / n
    out["itl_p50_ms"] = itls_ms[int(n * 0.50)]
    out["itl_p95_ms"] = itls_ms[min(int(n * 0.95), n - 1)]
    out["itl_max_ms"] = itls_ms[-1]
    # TPOT: mean seconds per output token across the generation window
    denom = max(completion_tokens - 1, 1) if completion_tokens else max(len(offs) - 1, 1)
    out["tpot_s"] = generation_s / denom
    return out


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
