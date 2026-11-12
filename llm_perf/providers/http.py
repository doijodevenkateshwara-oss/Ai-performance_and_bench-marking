"""Generic HTTP provider — Jinja2 request template + JSON-path response mapping."""
from __future__ import annotations

import json
import re
import time

from jinja2 import Template

from .base import NormalizedResponse, RequestSpec


def _jsonpath(obj, path: str):
    """Tiny JSONPath subset: dot keys + [n] indices, e.g. 'choices.0.message.content'."""
    cur = obj
    for part in re.split(r"\.|\[|\]", path):
        if part == "":
            continue
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


class HttpAdapter:
    name = "http"

    def default_base_url(self):
        return None

    def default_endpoint(self, stream: bool) -> str:
        return ""

    def build_request(self, spec: RequestSpec):
        headers = {"Content-Type": "application/json"}
        if spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        headers.update(spec.extra.get("headers", {}))
        template = spec.extra.get("request_template")
        if template:
            body = Template(template).render(prompt=spec.prompt, **spec.params)
            payload = body.encode() if isinstance(body, str) else body
        else:
            payload = json.dumps({"prompt": spec.prompt, **spec.params}).encode()
        base = (spec.base_url or "").rstrip("/")
        ep = spec.endpoint or spec.extra.get("endpoint") or ""
        url = f"{base}/{ep.lstrip('/')}" if ep else base
        return url, headers, payload

    def parse_response(self, status, headers, body, stream_iter=None) -> NormalizedResponse:
        start = time.monotonic()
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return NormalizedResponse(text="", raw={"error": "invalid json", "status": status})
        mapping = self._current_mapping or {}
        text = _jsonpath(obj, mapping.get("text", "text"))
        if text is None:
            # fallback: try common keys
            for k in ("text", "response", "output", "content", "choices"):
                v = obj.get(k)
                if isinstance(v, str):
                    text = v
                    break
                if isinstance(v, list) and v:
                    text = v[0].get("text") or v[0].get("message", {}).get("content") or ""
                    if text:
                        break
        if text is None:
            text = ""
        usage_obj = _jsonpath(obj, mapping.get("usage", "usage")) or {}
        if isinstance(usage_obj, dict):
            p = int(usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0)
            c = int(usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0)
        else:
            p = c = 0
        model = _jsonpath(obj, mapping.get("model", "model"))
        return NormalizedResponse(
            text=str(text), prompt_tokens=p, completion_tokens=c, total_tokens=p + c,
            model=str(model) if model else None, raw=obj,
            stage_metrics={"e2e_s": time.monotonic() - start})

    _current_mapping: dict = {}

    def with_mapping(self, mapping: dict) -> "HttpAdapter":
        self._current_mapping = mapping or {}
        return self
