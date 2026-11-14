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
from ..resilience import OperationFailed, execute
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

    def _http_once(
        self,
        method: str,
        url: str,
        headers: dict,
        json_body: dict | None,
        data: bytes | None,
    ):
        """One HTTP attempt; non-2xx/3xx statuses become classified failures."""
        resp = self.http.request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            data=data,
            stream=True,
        )
        status = resp.status_code
        if status >= 400:
            resp.close()
            raise OperationFailed(
                "http_5xx" if status >= 500 else "http_4xx",
                f"HTTP {status}",
            )
        return resp

    def _fallback_url(self, endpoint: str) -> str:
        """Resolve an alternate endpoint as absolute URL or path on base_url."""
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        return urljoin(self.target.base_url.rstrip("/") + "/", endpoint.lstrip("/"))

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
        res = t.resilience

        def call_grpc(host: str) -> NormalizedResponse:
            resp, _elapsed, err = self.grpc.call(host, spec)
            if err or resp is None:
                raise OperationFailed("exception", err or "no response")
            return resp

        def grpc_failure(resp: NormalizedResponse):
            err = resp.raw.get("error") if isinstance(resp.raw, dict) else None
            return ("llm_error", str(err)) if err else None

        def fallback(exc):
            fb = res.fallback
            if fb.type == "static_response":
                return NormalizedResponse(text=fb.response or "", raw={"fallback": "static_response"})
            if fb.type == "alternate_endpoint" and fb.endpoint:
                resp = call_grpc(fb.endpoint)
                failure = grpc_failure(resp)
                if failure:
                    raise OperationFailed(failure[0], failure[1], resp)
                return resp
            raise exc

        with request_span(scenario_name, t.name, "app_endpoint_grpc", stream=False) as span:
            start = time.monotonic()
            try:
                outcome = execute(
                    scenario=scenario_name,
                    target=t.name,
                    site="app_endpoint_grpc",
                    resilience=res,
                    operation=lambda: call_grpc(t.base_url),
                    result_failure=grpc_failure,
                    fallback=fallback,
                )
                resp = outcome.value
            except Exception as exc:
                elapsed = time.monotonic() - start
                record_error(scenario_name, "app_endpoint_grpc", str(exc), elapsed)
                log_request_event(scenario_name, elapsed * 1000, exception=str(exc))
                span.set_attribute("error", True)
                return LayerResult(error=str(exc), row=row, duration_s=elapsed)
            e2e = time.monotonic() - start
            span.set_attribute("llm_perf.degraded", outcome.degraded)
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
                duration_s=e2e,
                degraded=outcome.degraded,
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

        res = t.resilience

        def fallback(exc):
            fb = res.fallback
            if fb.type == "static_response":
                return NormalizedResponse(
                    text=fb.response or "",
                    raw={"fallback": "static_response"},
                )
            if fb.type == "alternate_endpoint" and fb.endpoint:
                return self._http_once(
                    t.method,
                    self._fallback_url(fb.endpoint),
                    headers,
                    body if raw is None else None,
                    raw,
                )
            raise exc

        with request_span(scenario_name, t.name, "app_endpoint", stream=False) as span:
            start = time.monotonic()
            try:
                outcome = execute(
                    scenario=scenario_name,
                    target=t.name,
                    site="app_endpoint",
                    resilience=res,
                    operation=lambda: self._http_once(
                        t.method,
                        url,
                        headers,
                        body if raw is None else None,
                        raw,
                    ),
                    fallback=fallback,
                )
                value = outcome.value
            except Exception as exc:
                e2e = time.monotonic() - start
                record_error(scenario_name, "app_endpoint", str(exc), e2e)
                log_request_event(scenario_name, e2e * 1000, exception=str(exc))
                span.set_attribute("error", True)
                return LayerResult(error=str(exc), row=row, duration_s=e2e)
            e2e = time.monotonic() - start
            span.set_attribute("llm_perf.degraded", outcome.degraded)
            if isinstance(value, NormalizedResponse):
                text = value.text or ""
                nr = value
                status = (value.raw or {}).get("status") if isinstance(value.raw, dict) else None
            else:
                # capped read: endpoint is untrusted, don't buffer unbounded bodies
                from ..security import read_response_text

                text = read_response_text(value)[:20000]
                status = value.status_code
                nr = NormalizedResponse(text=text, raw={"status": status})
            # reuse generic recording path without token accounting
            from ..metrics import instruments

            instruments()["request_total"].add(
                1, {"scenario": scenario_name, "provider": "app_endpoint", "result": "success"}
            )
            instruments()["request_duration"].record(
                e2e, {"scenario": scenario_name, "provider": "app_endpoint"}
            )
            log_request_event(scenario_name, e2e * 1000)
            if status is not None:
                span.set_attribute("http.status_code", status)
            return LayerResult(
                output_text=text,
                response=nr,
                row=row,
                duration_s=e2e,
                degraded=outcome.degraded,
            )
