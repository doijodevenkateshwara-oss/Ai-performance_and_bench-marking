"""Provider adapter protocol and normalized response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class NormalizedResponse:
    """Uniform response shape across providers."""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    model: str | None = None
    raw: Any = None
    ttft_s: float | None = None  # set for streaming responses
    # decode-phase timing (streaming only; None for unary calls)
    tpot_s: float | None = None  # time per output token
    generation_s: float | None = None  # first token -> last token
    stream_stats: dict = field(default_factory=dict)  # itl_avg/p50/p95/max_ms, chunks
    queue_wait_s: float | None = None  # task start -> request dispatch
    retries: int = 0  # retry attempts made for this call
    timed_out: bool = False
    stage_metrics: dict[str, float] = field(default_factory=dict)
    tool_calls: list[dict] | None = None  # OpenAI-style tool calls from the model

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
    stream_mode: str = "full"  # full | first_token
    api_key: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    timeout_s: float = 120.0
    messages: list[dict] | None = None  # multi-turn override (agent loops)
    extra: dict[str, Any] = field(default_factory=dict)  # azure/aws/grpc config


class ProviderAdapter(Protocol):
    """Builds and parses provider-specific requests."""

    name: str

    def default_base_url(self) -> str | None: ...

    def default_endpoint(self, stream: bool) -> str: ...

    def build_request(self, spec: RequestSpec) -> tuple[str, dict, bytes | None]:
        """Return (url, headers, json_body_or_None)."""
        ...

    def parse_response(
        self, status: int, headers: dict, body: bytes, stream_iter=None
    ) -> NormalizedResponse: ...


def usage_tokens(usage: dict | None) -> tuple[int, int, int]:
    """Extract (prompt, completion, total) from an OpenAI-style usage dict."""
    if not usage:
        return 0, 0, 0
    p = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    c = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    t = int(usage.get("total_tokens") or (p + c))
    return p, c, t
