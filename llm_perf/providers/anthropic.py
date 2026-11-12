"""Anthropic Messages API provider."""

from __future__ import annotations

import json

from .base import NormalizedResponse, RequestSpec
from .http_common import anthropic_sse_extractor, auth_headers, stream_sse, stream_timing_stats


class AnthropicAdapter:
    name = "anthropic"

    def default_base_url(self):
        return "https://api.anthropic.com"

    def default_endpoint(self, stream: bool) -> str:
        return "/v1/messages"

    def build_request(self, spec: RequestSpec):
        headers = auth_headers(spec)
        headers["x-api-key"] = spec.api_key or ""
        headers["anthropic-version"] = "2023-06-01"
        headers.pop("Authorization", None)
        body = {
            "model": spec.params.get("model", "claude-sonnet-4-5"),
            "max_tokens": spec.params.get("max_tokens", spec.params.get("max_completion_tokens", 1024)),
            "messages": [{"role": "user", "content": spec.prompt}],
        }
        for k in ("temperature", "top_p", "top_k", "stop_sequences", "system", "tools", "tool_choice"):
            if k in spec.params:
                body[k] = spec.params[k]
        if spec.stream:
            body["stream"] = True
        base = (spec.base_url or self.default_base_url()).rstrip("/")
        return f"{base}{spec.endpoint or self.default_endpoint(spec.stream)}", headers, body

    def parse_response(
        self, status, headers, body, stream_iter=None, stream_mode: str = "full"
    ) -> NormalizedResponse:
        if stream_iter is not None:
            # Anthropic SSE includes `event:` lines; stream_sse skips non-"data:" lines.
            sr = stream_sse(
                stream_iter, anthropic_sse_extractor, first_token_only=(stream_mode == "first_token")
            )
            p = int((sr.usage or {}).get("input_tokens") or 0)
            c = int((sr.usage or {}).get("output_tokens") or 0)
            timing = stream_timing_stats(sr.chunk_offsets_s, c)
            return NormalizedResponse(
                text=sr.text,
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=p + c,
                ttft_s=sr.ttft_s,
                tpot_s=timing["tpot_s"],
                generation_s=timing["generation_s"],
                stream_stats=timing,
                raw=sr.usage,
            )
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return NormalizedResponse(text="", raw={"error": "invalid json", "status": status})
        if "error" in obj and "content" not in obj:
            return NormalizedResponse(text="", raw=obj)
        parts = []
        for block in obj.get("content", []) or []:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        usage = obj.get("usage") or {}
        p = int(usage.get("input_tokens") or 0)
        c = int(usage.get("output_tokens") or 0)
        return NormalizedResponse(
            text="".join(parts),
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=p + c,
            finish_reason=obj.get("stop_reason"),
            model=obj.get("model"),
            raw=obj,
        )
