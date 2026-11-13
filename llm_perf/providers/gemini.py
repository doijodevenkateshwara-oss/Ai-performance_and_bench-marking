"""Google Gemini API provider (AI Studio Generative Language API).

Supports generateContent (unary) and streamGenerateContent?alt=sse.
The same request shape works for Vertex AI by overriding `endpoint`
with a path containing {model}, e.g.
  /v1/projects/<p>/locations/<l>/publishers/google/models/{model}:generateContent

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json

from .base import NormalizedResponse, RequestSpec
from .http_common import stream_sse, stream_timing_stats

# params keys mapped into generationConfig (snake_case -> camelCase)
_GEN_MAP = {
    "max_tokens": "maxOutputTokens",
    "max_completion_tokens": "maxOutputTokens",
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
    "candidate_count": "candidateCount",
    "stop": "stopSequences",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "seed": "seed",
    "response_mime_type": "responseMimeType",
    "response_schema": "responseSchema",
}


def _gemini_usage(usage_metadata: dict | None) -> dict | None:
    """Normalize usageMetadata to OpenAI-style token keys (None-safe)."""
    if not usage_metadata:
        return None
    p = int(usage_metadata.get("promptTokenCount") or 0)
    c = int(usage_metadata.get("candidatesTokenCount") or 0)
    t = int(usage_metadata.get("totalTokenCount") or (p + c))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


def gemini_sse_extractor(payload: str):
    """Extract (text_delta, usage, done) from a streamGenerateContent chunk."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return "", None, False
    cands = obj.get("candidates") or []
    text = ""
    finish = None
    if cands:
        parts = ((cands[0].get("content") or {}).get("parts")) or []
        # thought parts (Gemini 2.5 thinking) are not user-visible output
        text = "".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text") and not p.get("thought")
        )
        finish = cands[0].get("finishReason")
    return text, _gemini_usage(obj.get("usageMetadata")), bool(finish)


class GeminiAdapter:
    """Google Gemini generateContent / streamGenerateContent adapter."""

    name = "gemini"

    def default_base_url(self):
        """Default AI Studio base URL."""
        return "https://generativelanguage.googleapis.com"

    def default_endpoint(self, stream: bool) -> str:
        """Endpoint template; {model} and {version} are substituted."""
        method = "streamGenerateContent" if stream else "generateContent"
        return "/{version}/models/{model}:" + method + ("?alt=sse" if stream else "")

    def _endpoint(self, spec: RequestSpec) -> str:
        """Resolve endpoint path, substituting model/version placeholders."""
        model = spec.params.get("model", "gemini-2.0-flash")
        version = (spec.extra.get("gemini") or {}).get("api_version", "v1beta")
        tpl = spec.endpoint or self.default_endpoint(spec.stream)
        return tpl.replace("{model}", model).replace("{version}", version)

    def _contents(self, spec: RequestSpec) -> list[dict]:
        """Build Gemini contents from spec.messages or the single prompt."""
        if spec.messages:
            contents = []
            for m in spec.messages:
                role = "model" if m.get("role") in ("assistant", "model") else "user"
                text = m.get("content") or ""
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                contents.append({"role": role, "parts": [{"text": text}]})
            return contents
        return [{"role": "user", "parts": [{"text": spec.prompt}]}]

    def build_request(self, spec: RequestSpec):
        """Assemble URL, x-goog-api-key headers, and generateContent body."""
        headers = {"Content-Type": "application/json"}
        if spec.api_key:
            headers["x-goog-api-key"] = spec.api_key
        body: dict = {"contents": self._contents(spec)}
        if spec.params.get("system"):
            body["systemInstruction"] = {"parts": [{"text": spec.params["system"]}]}
        gen: dict = {}
        for k, gk in _GEN_MAP.items():
            if k in spec.params:
                gen[gk] = spec.params[k]
        extra_gen = spec.params.get("generation_config")
        if isinstance(extra_gen, dict):
            gen.update(extra_gen)
        if gen:
            body["generationConfig"] = gen
        if spec.params.get("safety_settings"):
            body["safetySettings"] = spec.params["safety_settings"]
        url = f"{(spec.base_url or self.default_base_url()).rstrip('/')}{self._endpoint(spec)}"
        return url, headers, body

    def parse_response(
        self, status, headers, body, stream_iter=None, stream_mode: str = "full"
    ) -> NormalizedResponse:
        """Parse unary or SSE Gemini responses into NormalizedResponse."""
        if stream_iter is not None:
            sr = stream_sse(
                stream_iter, gemini_sse_extractor, first_token_only=(stream_mode == "first_token")
            )
            usage = sr.usage or {}
            p, c, t = (
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                int(usage.get("total_tokens") or 0),
            )
            timing = stream_timing_stats(sr.chunk_offsets_s, c)
            return NormalizedResponse(
                text=sr.text,
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=t or p + c,
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
        if "error" in obj:
            return NormalizedResponse(text="", raw=obj)
        cands = obj.get("candidates") or []
        text = ""
        finish = None
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            text = "".join(
                p.get("text", "")
                for p in parts
                if isinstance(p, dict) and p.get("text") and not p.get("thought")
            )
            finish = cands[0].get("finishReason")
        usage = _gemini_usage(obj.get("usageMetadata")) or {}
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or (p + c))
        return NormalizedResponse(
            text=text,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=t,
            finish_reason=finish,
            model=obj.get("model"),
            raw=obj,
        )
