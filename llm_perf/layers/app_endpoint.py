"""Layer 2c: black-box app endpoint (HTTP/gRPC) testing."""

from __future__ import annotations

import os
import time
from urllib.parse import urljoin

from jinja2 import Template

from ..config import AppEndpointTarget
from ..metrics import record_error, request_span
from ..prompts import PromptPicker
from ..providers.base import NormalizedResponse
from ..transports.http import HttpTransport
from .base import LayerResult


class AppEndpointLayer:
    def __init__(self, target: AppEndpointTarget, picker: PromptPicker):
        self.target = target
        self.picker = picker
        self.http = HttpTransport(timeout_s=target.timeout_s)

    def _build_body(self, prompt: str, row: dict | None) -> tuple[dict, bytes | None]:
        t = self.target
        ctx = {"prompt": prompt, "row": row or {}}
        if row:
            ctx.update({k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))})
        if t.request_template:
            rendered = Template(t.request_template).render(**ctx)
            return {}, rendered.encode()
        return {"prompt": prompt, **(row or {})}, None

    def run(
        self, prompt: str, row: dict | None, scenario_params: dict, scenario_name: str = "default"
    ) -> LayerResult:
        t = self.target
        headers = dict(t.headers)
        if t.api_key_env and os.environ.get(t.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[t.api_key_env]}"
        url = urljoin(t.base_url.rstrip("/") + "/", (t.endpoint or "").lstrip("/"))
        body, raw = self._build_body(prompt, row)

        with request_span(scenario_name, t.name, "app_endpoint", stream=False) as span:
            start = time.monotonic()
            resp, elapsed, err = self.http.call_timed(
                t.method, url, headers=headers, json_body=body if raw is None else None, data=raw
            )
            if err or resp is None:
                record_error(scenario_name, "app_endpoint", err or "no response", elapsed)
                from ..metrics import log_request_event

                log_request_event(scenario_name, elapsed * 1000, exception=err or "no response")
                span.set_attribute("error", True)
                return LayerResult(error=err or "request failed", row=row, stage_metrics={"e2e_s": elapsed})
            text = resp.text[:20000]
            e2e = time.monotonic() - start
            nr = NormalizedResponse(text=text, raw={"status": resp.status_code})
            if resp.status_code >= 400:
                record_error(scenario_name, "app_endpoint", f"HTTP {resp.status_code}: {text[:200]}", e2e)
                from ..metrics import log_request_event

                log_request_event(scenario_name, e2e * 1000, exception=f"HTTP {resp.status_code}")
                span.set_attribute("error", True)
                return LayerResult(
                    error=f"HTTP {resp.status_code}",
                    response=nr,
                    output_text=text,
                    row=row,
                    stage_metrics={"e2e_s": e2e},
                )
            # reuse generic recording path without token accounting
            from ..metrics import instruments, log_request_event

            instruments()["request_total"].add(
                1, {"scenario": scenario_name, "provider": "app_endpoint", "result": "success"}
            )
            instruments()["request_duration"].record(
                e2e, {"scenario": scenario_name, "provider": "app_endpoint"}
            )
            log_request_event(scenario_name, e2e * 1000)
            span.set_attribute("http.status_code", resp.status_code)
            return LayerResult(output_text=text, response=nr, row=row, stage_metrics={"e2e_s": e2e})
