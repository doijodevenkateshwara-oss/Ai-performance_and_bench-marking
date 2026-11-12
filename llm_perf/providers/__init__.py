"""Provider adapter registry."""

from __future__ import annotations

from .anthropic import AnthropicAdapter
from .azure import AzureAdapter
from .base import NormalizedResponse, ProviderAdapter, RequestSpec, usage_tokens
from .bedrock import BedrockAdapter
from .grpc import GrpcAdapter
from .http import HttpAdapter
from .openai import OpenAIAdapter

_ADAPTERS: dict[str, ProviderAdapter] = {}


def get_adapter(provider: str, **kwargs) -> ProviderAdapter:
    key = f"openai:{kwargs['api_style']}" if provider == "openai" and kwargs.get("api_style") else provider
    if key not in _ADAPTERS:
        if provider == "openai":
            _ADAPTERS[key] = OpenAIAdapter(api_style=kwargs.get("api_style", "chat"))
        elif provider == "azure":
            _ADAPTERS[key] = AzureAdapter(api_style=kwargs.get("api_style", "chat"))
        elif provider == "anthropic":
            _ADAPTERS[key] = AnthropicAdapter()
        elif provider == "bedrock":
            _ADAPTERS[key] = BedrockAdapter()
        elif provider == "grpc":
            _ADAPTERS[key] = GrpcAdapter()
        elif provider == "http":
            _ADAPTERS[key] = HttpAdapter()
        else:
            raise ValueError(f"unknown provider: {provider}")
    return _ADAPTERS[key]


def call_llm(spec: RequestSpec) -> NormalizedResponse:
    """Unified entry point: dispatch to adapter (handles bedrock/grpc callables)."""
    adapter = get_adapter(spec.provider)
    if spec.provider == "bedrock":
        return adapter.call(spec)  # type: ignore[attr-defined]
    if spec.provider == "grpc":
        return adapter.call(spec)  # type: ignore[attr-defined]

    # http adapter is cheap to construct; fresh instance keeps response_mapping safe
    if spec.provider == "http":
        adapter = HttpAdapter().with_mapping(spec.extra.get("response_mapping") or {})
    url, headers, body = adapter.build_request(spec)
    from .http_common import HttpClient

    client = HttpClient(timeout_s=spec.timeout_s)
    if spec.stream and spec.provider in ("openai", "azure", "anthropic"):
        resp = client.post(url, headers, body, stream=True, timeout=spec.timeout_s)
        if resp.status_code >= 400:
            return NormalizedResponse(text="", raw={"error": resp.text[:2000], "status": resp.status_code})
        return adapter.parse_response(
            resp.status_code,
            dict(resp.headers),
            b"",
            stream_iter=resp.iter_lines(),
            stream_mode=spec.stream_mode,
        )
    resp = client.post(url, headers, body, stream=False, timeout=spec.timeout_s)
    if resp.status_code >= 400:
        return NormalizedResponse(text="", raw={"error": resp.text[:2000], "status": resp.status_code})
    return adapter.parse_response(resp.status_code, dict(resp.headers), resp.content)


__all__ = [
    "NormalizedResponse",
    "ProviderAdapter",
    "RequestSpec",
    "call_llm",
    "get_adapter",
    "usage_tokens",
]
