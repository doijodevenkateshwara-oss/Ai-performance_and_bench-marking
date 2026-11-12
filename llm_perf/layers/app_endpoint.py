"""Layer 2c: black-box app endpoint (HTTP/gRPC) testing.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import os
import time
from urllib.parse import urljoin

from jinja2 import Template

from ..config import AppEndpointTarget
from ..metrics import log_request_event, record_error, request_span
from ..prompts import PromptPicker
from ..providers.base import NormalizedResponse, RequestSpec
from ..transports.grpc import GrpcTransport
from ..transports.http import HttpTransport
from .base import LayerResult


class AppEndpointLayer:
    """Runs black-box HTTP or gRPC application endpoints under load."""

    def __init__(self, target: AppEndpointTarget, picker: PromptPicker):
        """AppEndpointLayer(target, picker): pre-builds HTTP/gRPC transports."""
        self.target = target
        self.picker = picker
        self.http = HttpTransport(timeout_s=target.timeout_s)
        self.grpc = GrpcTransport(timeout_s=target.timeout_s) if target.transport == "grpc" else None

    def _build_body(self, prompt: str, row: dict | None) -> tuple[dict, bytes | None]:
        """Render the request body from prompt/row/template."""
        t = self.target
        ctx = {"prompt": prompt, "row": row or {}}
        if row:
            ctx.update({k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))})
        if t.request_template:
            rendered = Template(t.request_template).render(**ctx)
            return {}, rendered.encode()
        return {"prompt": prompt, **(row or {})}, None

    def _run_grpc(
        self, prompt: str, row: dict | None, scenario_params: dict, scenario_name: str
    ) -> LayerResult:
        """gRPC path: reuse the provider gRPC machinery against a raw endpoint."""
        t = self.target
        params = {k: v for k, v in scenario_params.items() if not k.startswith("_")}
        spec = RequestSpec(
            provider="grpc",
            prompt=prompt,
            params=params,
            base_url=t.base_url,
            timeout_s=t.timeout_s,
            extra={"grpc": t.grpc.model_dump() if t.grpc else {}},
        )
        with request_span(scenario_name, t.name, "app_endpoint_grpc", stream=False) as span:
            resp, elapsed, err = self.grpc.call(t.base_url, spec)
            if err or resp is None:
                record_error(scenario_name, "app_endpoint_grpc", err or "no response", elapsed)
                log_request_event(scenario_name, elapsed * 1000, exception=err or "no response")
                span.set_attribute("error", True)
                return LayerResult(
                    error=err or "request failed",
                    row=row,
                )
            grpc_err = resp.raw.get("error") if isinstance(resp.raw, dict) else None
            if grpc_err:
                record_error(scenario_name, "app_endpoint_grpc", str(grpc_err), elapsed)
                log_request_event(scenario_name, elapsed * 1000, exception=str(grpc_err))
                span.set_attribute("error", True)
                return LayerResult(
                    error=str(grpc_err),
                    response=resp,
                    row=row,
                )
            e2e = elapsed  # GrpcTransport.call already measured the round trip
            from ..metrics import instruments

            instruments()["request_total"].add(
                1, {"scenario": scenario_name, "provider": "app_endpoint_grpc", "result": "success"}
            )
            instruments()["request_duration"].record(
                e2e, {"scenario": scenario_name, "provider": "app_endpoint_grpc"}
            )
            log_request_event(scenario_name, e2e * 1000)
            span.set_attribute("llm_perf.output_chars", len(resp.text or ""))
            return LayerResult(
                output_text=resp.text,
                response=resp,
                row=row,
            )

    def run(
        self, prompt: str, row: dict | None, scenario_params: dict, scenario_name: str = "default"
    ) -> LayerResult:
        """Execute one request against the endpoint and record metrics."""
        t = self.target
        if t.transport == "grpc":
            return self._run_grpc(prompt, row, scenario_params, scenario_name)

        headers = dict(t.headers)
        if t.api_key_env and os.environ.get(t.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[t.api_key_env]}"
        url = urljoin(t.base_url.rstrip("/") + "/", (t.endpoint or "").lstrip("/"))
        body, raw = self._build_body(prompt, row)

        with request_span(scenario_name, t.name, "app_endpoint", stream=False) as span:
            start = time.monotonic()
            resp, elapsed, err = self.http.call_timed(
                t.method,
                url,
                headers=headers,
                json_body=body if raw is None else None,
                data=raw,
                stream=True,  # stream so read_response_text can cap the download
            )
            if err or resp is None:
                record_error(scenario_name, "app_endpoint", err or "no response", elapsed)
                log_request_event(scenario_name, elapsed * 1000, exception=err or "no response")
                span.set_attribute("error", True)
                return LayerResult(
                    error=err or "request failed",
                    row=row,
                )
            e2e = time.monotonic() - start
            if resp.status_code >= 400:
                resp.close()  # release the connection without buffering the error body
                record_error(scenario_name, "app_endpoint", f"HTTP {resp.status_code}", e2e)
                log_request_event(scenario_name, e2e * 1000, exception=f"HTTP {resp.status_code}")
                span.set_attribute("error", True)
                nr = NormalizedResponse(text="", raw={"status": resp.status_code})
                return LayerResult(
                    error=f"HTTP {resp.status_code}",
                    response=nr,
                    row=row,
                )
            # capped read: endpoint is untrusted, don't buffer unbounded bodies
            from ..security import read_response_text

            text = read_response_text(resp)[:20000]
            nr = NormalizedResponse(text=text, raw={"status": resp.status_code})
            # reuse generic recording path without token accounting
            from ..metrics import instruments

            instruments()["request_total"].add(
                1, {"scenario": scenario_name, "provider": "app_endpoint", "result": "success"}
            )
            instruments()["request_duration"].record(
                e2e, {"scenario": scenario_name, "provider": "app_endpoint"}
            )
            log_request_event(scenario_name, e2e * 1000)
            span.set_attribute("http.status_code", resp.status_code)
            return LayerResult(
                output_text=text,
                response=nr,
                row=row,
            )
