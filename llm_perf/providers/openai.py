"""OpenAI-compatible provider (chat completions, responses API, embeddings).

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json

from .base import NormalizedResponse, RequestSpec, usage_tokens
from .http_common import auth_headers, openai_sse_extractor, stream_sse, stream_timing_stats


class OpenAIAdapter:
    """OpenAI chat-completions / responses-API provider adapter."""

    name = "openai"

    def __init__(self, api_style: str = "chat"):
        # api_style: "chat" (chat/completions) or "responses"
        """OpenAIAdapter(api_style): 'chat' or 'responses'."""
        self.api_style = api_style

    def default_base_url(self):
        """Default OpenAI API base URL."""
        return "https://api.openai.com/v1"

    def default_endpoint(self, stream: bool) -> str:
        """Endpoint path for the configured api_style."""
        return "/chat/completions" if self.api_style == "chat" else "/responses"

    def build_request(self, spec: RequestSpec):
        """Assemble chat/responses body with supported param passthrough."""
        headers = auth_headers(spec)
        model = spec.params.get("model", "gpt-4o-mini")
        if self.api_style == "chat":
            messages = spec.messages or [{"role": "user", "content": spec.prompt}]
            body = {
                "model": model,
                "messages": messages,
            }
            for k in (
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "seed",
                "tools",
                "tool_choice",
                "response_format",
                "n",
            ):
                if k in spec.params:
                    body[k] = spec.params[k]
            if spec.stream:
                body["stream"] = True
                # request usage in the final chunk so token counts survive streaming
                body.setdefault("stream_options", {"include_usage": True})
        else:  # responses API
            body = {"model": model, "input": spec.prompt}
            for k in ("max_output_tokens", "temperature", "top_p", "tools", "text"):
                if k in spec.params:
                    body[k] = spec.params[k]
            if spec.stream:
                body["stream"] = True
        return (
            f"{(spec.base_url or self.default_base_url()).rstrip('/')}{spec.endpoint or self.default_endpoint(spec.stream)}",
            headers,
            body,
        )

    def parse_response(
        self, status, headers, body, stream_iter=None, stream_mode: str = "full"
    ) -> NormalizedResponse:
        """Parse unary or SSE OpenAI responses into NormalizedResponse."""
        if stream_iter is not None:
            sr = stream_sse(
                stream_iter, openai_sse_extractor, first_token_only=(stream_mode == "first_token")
            )
            p, c, t = usage_tokens(sr.usage)
            timing = stream_timing_stats(sr.chunk_offsets_s, c)
            return NormalizedResponse(
                text=sr.text,
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=t,
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
        if self.api_style == "responses":
            text = ""
            for item in obj.get("output", []) or []:
                for c in item.get("content", []) or []:
                    if c.get("type") in ("output_text", "text"):
                        text += c.get("text", "")
            usage = obj.get("usage") or {}
            p = int(usage.get("input_tokens") or 0)
            c = int(usage.get("output_tokens") or 0)
            return NormalizedResponse(
                text=text,
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=p + c,
                finish_reason=obj.get("status"),
                model=obj.get("model"),
                raw=obj,
            )
        choice = (obj.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        p, c, t = usage_tokens(obj.get("usage"))
        return NormalizedResponse(
            text=text,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=t,
            finish_reason=choice.get("finish_reason"),
            model=obj.get("model"),
            raw=obj,
            tool_calls=msg.get("tool_calls"),
        )
