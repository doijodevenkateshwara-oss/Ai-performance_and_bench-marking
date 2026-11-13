"""Provider adapter registry.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import random
import time

import requests

from .anthropic import AnthropicAdapter
from .azure import AzureAdapter
from .base import NormalizedResponse, ProviderAdapter, RequestSpec, usage_tokens
from .bedrock import BedrockAdapter
from .gemini import GeminiAdapter
from .grpc import GrpcAdapter
from .http import HttpAdapter
from .openai import OpenAIAdapter

_ADAPTERS: dict[str, ProviderAdapter] = {}

# exceptions worth retrying (in addition to HTTP 5xx when enabled)
_RETRYABLE_EXC = (requests.Timeout, requests.ConnectionError)


def get_adapter(provider: str, **kwargs) -> ProviderAdapter:
    # cache per provider+variant so mixed api_style targets don't collide
    """Return the cached adapter for a provider (+api_style variant)."""
    variant = kwargs.get("api_style") if provider in ("openai", "azure") else None
    key = f"{provider}:{variant}" if variant else provider
    if key not in _ADAPTERS:
        if provider == "openai":
            _ADAPTERS[key] = OpenAIAdapter(api_style=kwargs.get("api_style", "chat"))
        elif provider == "azure":
            _ADAPTERS[key] = AzureAdapter(api_style=kwargs.get("api_style", "chat"))
        elif provider == "anthropic":
            _ADAPTERS[key] = AnthropicAdapter()
        elif provider == "bedrock":
            _ADAPTERS[key] = BedrockAdapter()
        elif provider == "gemini":
            _ADAPTERS[key] = GeminiAdapter()
        elif provider == "grpc":
            _ADAPTERS[key] = GrpcAdapter()
        elif provider == "http":
            _ADAPTERS[key] = HttpAdapter()
        else:
            raise ValueError(f"unknown provider: {provider}")
    return _ADAPTERS[key]


def _queue_wait_s(spec: RequestSpec) -> float | None:
    """Seconds from virtual-user task start to request dispatch."""
    t0 = spec.params.get("_task_start")
    if not isinstance(t0, (int, float)):
        return None
    return max(time.monotonic() - t0, 0.0)


def _backoff_sleep(base_s: float, attempt: int) -> None:
    """Sleep with exponential backoff plus jitter between retries."""
    time.sleep(base_s * (2 ** (attempt - 1)) + random.uniform(0, base_s * 0.25))


def _post_with_retries(client, url, headers, body, stream, timeout, retry_cfg):
    """POST with optional retries. Returns (response|None, error|None, retries, timed_out)."""
    max_retries = int((retry_cfg or {}).get("max_retries", 0) or 0)
    backoff = float((retry_cfg or {}).get("backoff_s", 0.5) or 0.0)
    retry_on = set((retry_cfg or {}).get("retry_on") or ["timeout", "connect", "5xx"])
    retries = 0
    timed_out = False
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.post(url, headers, body, stream=stream, timeout=timeout)
        except _RETRYABLE_EXC as e:
            last_err = e
            timed_out = isinstance(e, requests.Timeout)
            kind = "timeout" if timed_out else "connect"
            if kind in retry_on and attempt < max_retries:
                retries += 1
                if backoff:
                    _backoff_sleep(backoff, attempt + 1)
                continue
            return None, str(e), retries, timed_out
        except requests.RequestException as e:
            return None, str(e), retries, False
        if resp.status_code >= 500 and "5xx" in retry_on and attempt < max_retries:
            retries += 1
            resp.close()  # release the connection before retrying
            if backoff:
                _backoff_sleep(backoff, attempt + 1)
            continue
        return resp, None, retries, False
    return None, last_err or "retries exhausted", retries, timed_out


def call_llm(spec: RequestSpec) -> NormalizedResponse:
    """Unified entry point: dispatch to adapter (handles bedrock/grpc callables)."""
    adapter = get_adapter(spec.provider)
    if spec.provider == "bedrock":
        resp = adapter.call(spec)  # type: ignore[attr-defined]
        resp.queue_wait_s = _queue_wait_s(spec)
        return resp
    if spec.provider == "grpc":
        resp = adapter.call(spec)  # type: ignore[attr-defined]
        resp.queue_wait_s = _queue_wait_s(spec)
        return resp

    # http adapter is cheap to construct; fresh instance keeps response_mapping safe
    if spec.provider == "http":
        adapter = HttpAdapter().with_mapping(spec.extra.get("response_mapping") or {})
    url, headers, body = adapter.build_request(spec)
    queue_wait = _queue_wait_s(spec)
    from .http_common import HttpClient

    client = HttpClient(timeout_s=spec.timeout_s)
    retry_cfg = spec.extra.get("retry") or {}
    want_stream = spec.stream and spec.provider in ("openai", "azure", "anthropic", "gemini")
    resp, err, retries, timed_out = _post_with_retries(
        client,
        url,
        headers,
        body,
        stream=want_stream,
        timeout=spec.timeout_s,
        retry_cfg=retry_cfg,
    )
    if err is not None or resp is None:
        return NormalizedResponse(
            text="",
            raw={"error": err or "no response"},
            queue_wait_s=queue_wait,
            retries=retries,
            timed_out=timed_out,
        )
    if resp.status_code >= 400:
        nr = NormalizedResponse(
            text="",
            raw={"error": resp.text[:2000], "status": resp.status_code},
            queue_wait_s=queue_wait,
            retries=retries,
            timed_out=timed_out,
        )
        return nr
    if want_stream:
        nr = adapter.parse_response(
            resp.status_code,
            dict(resp.headers),
            b"",
            stream_iter=resp.iter_lines(),
            stream_mode=spec.stream_mode,
        )
    else:
        nr = adapter.parse_response(resp.status_code, dict(resp.headers), resp.content)
    nr.queue_wait_s = queue_wait
    nr.retries = retries
    nr.timed_out = timed_out
    return nr


__all__ = [
    "NormalizedResponse",
    "ProviderAdapter",
    "RequestSpec",
    "call_llm",
    "get_adapter",
    "usage_tokens",
]
