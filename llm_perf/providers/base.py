"""Provider adapter protocol and normalized response types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class NormalizedResponse:
    """Uniform response shape across providers."""
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    raw: Any = None
    ttft_s: Optional[float] = None          # set for streaming responses
    stage_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class RequestSpec:
    """Everything needed to execute one LLM call."""
    provider: str
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    stream_mode: str = "full"               # full | first_token
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    endpoint: Optional[str] = None
    timeout_s: float = 120.0
    extra: dict[str, Any] = field(default_factory=dict)  # azure/aws/grpc config


class ProviderAdapter(Protocol):
    """Builds and parses provider-specific requests."""

    name: str

    def default_base_url(self) -> Optional[str]:
        ...

    def default_endpoint(self, stream: bool) -> str:
        ...

    def build_request(self, spec: RequestSpec) -> tuple[str, dict, bytes | None]:
        """Return (url, headers, json_body_or_None)."""
        ...

    def parse_response(self, status: int, headers: dict, body: bytes,
                       stream_iter=None) -> NormalizedResponse:
        ...


def usage_tokens(usage: dict | None) -> tuple[int, int, int]:
    """Extract (prompt, completion, total) from an OpenAI-style usage dict."""
    if not usage:
        return 0, 0, 0
    p = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    c = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    t = int(usage.get("total_tokens") or (p + c))
    return p, c, t
