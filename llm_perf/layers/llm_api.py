"""Layer 1: direct LLM API calls via provider adapters.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import os
import time
from functools import partial

from ..config import LlmApiTarget
from ..metrics import log_request_event, record_error, record_response, request_span
from ..prompts import PromptPicker
from ..providers import NormalizedResponse, RequestSpec, call_llm
from ..resilience import OperationFailed, execute, provider_retry_config
from .base import LayerResult


def _llm_failure(resp: NormalizedResponse) -> tuple[str, str] | None:
    """Classify a normalized provider result for the resilience engine."""
    raw = resp.raw if isinstance(resp.raw, dict) else {}
    err = raw.get("error")
    if not err:
        return None
    status = int(raw.get("status") or 0)
    if resp.timed_out:
        return "timeout", str(err)
    if status >= 500:
        return "http_5xx", str(err)
    if status >= 400:
        return "http_4xx", str(err)
    return "llm_error", str(err)


class LlmApiLayer:
    """Direct vendor-API layer: builds a RequestSpec and calls call_llm."""

    def __init__(self, target: LlmApiTarget, picker: PromptPicker):
        """LlmApiLayer(target, picker)."""
        self.target = target
        self.picker = picker

    def _alternate_spec(
        self,
        prompt: str,
        params: dict,
        spec: RequestSpec,
        api_key: str | None,
        cfg: dict,
    ) -> RequestSpec:
        """Build a provider request for an alternate-LLM fallback."""
        t = self.target
        alt_params = {**params, **(cfg.get("params") or {})}
        if cfg.get("model"):
            alt_params["model"] = cfg["model"]
        return RequestSpec(
            provider=cfg.get("provider", t.provider),
            prompt=prompt,
            params=alt_params,
            stream=spec.stream,
            stream_mode=t.stream_mode,
            api_key=os.environ.get(cfg.get("api_key_env", "")) or api_key,
            base_url=cfg.get("base_url", t.base_url),
            endpoint=cfg.get("endpoint", spec.endpoint),
            timeout_s=cfg.get("timeout_s", t.timeout_s),
            extra={
                "azure": cfg.get("azure") or spec.extra.get("azure") or {},
                "aws": cfg.get("aws") or spec.extra.get("aws") or {},
                "gemini": cfg.get("gemini") or spec.extra.get("gemini") or {},
                "grpc": cfg.get("grpc") or spec.extra.get("grpc") or {},
                "request_template": cfg.get("request_template", t.request_template),
                "response_mapping": cfg.get("response_mapping", t.response_mapping),
                "headers": {**t.tags.get("headers", {}), **(cfg.get("headers") or {})},
                "retry": spec.extra.get("retry") or {},
            },
        )

    def _fallback(
        self,
        exc: Exception,
        prompt: str,
        params: dict,
        spec: RequestSpec,
        api_key: str | None,
    ):
        """Execute the configured direct-LLM fallback."""
        fb = self.target.resilience.fallback
        if fb.type == "static_response":
            return NormalizedResponse(text=fb.response or "", raw={"fallback": "static_response"})
        if fb.type == "alternate_llm" and fb.llm:
            resp = call_llm(self._alternate_spec(prompt, params, spec, api_key, fb.llm))
            failure = _llm_failure(resp)
            if failure:
                raise OperationFailed(failure[0], failure[1], resp)
            return resp
        raise exc

    def run(
        self, prompt: str, row: dict | None, scenario_params: dict, scenario_name: str = "default"
    ) -> LayerResult:
        """Execute one LLM call, record metrics/traces, return LayerResult."""
        t = self.target
        params = {**t.params, **scenario_params}
        api_key = None
        if t.api_key_env:
            api_key = os.environ.get(t.api_key_env)
        spec = RequestSpec(
            provider=t.provider,
            prompt=prompt,
            params=params,
            stream=(scenario_params.get("_stream", t.stream)),
            stream_mode=t.stream_mode,
            api_key=api_key,
            base_url=t.base_url,
            endpoint=t.endpoint or (t.params.get("endpoint")),
            timeout_s=t.timeout_s,
            extra={
                "azure": t.azure.model_dump() if t.azure else {},
                "aws": t.aws.model_dump() if t.aws else {},
                "gemini": t.gemini.model_dump() if t.gemini else {},
                "grpc": t.grpc.model_dump() if t.grpc else {},
                "request_template": t.request_template,
                "response_mapping": t.response_mapping,
                "headers": t.tags.get("headers", {}),
                "retry": provider_retry_config(t.resilience.retry)
                or {"max_retries": t.retries, "backoff_s": t.retry_backoff_s},
            },
        )
        model = params.get("model") or params.get("model_id")
        res = t.resilience
        fallback = partial(
            self._fallback,
            prompt=prompt,
            params=params,
            spec=spec,
            api_key=api_key,
        )

        with request_span(scenario_name, t.name, t.provider, model=model, stream=spec.stream) as span:
            start = time.monotonic()
            try:
                outcome = execute(
                    scenario=scenario_name,
                    target=t.name,
                    site="llm_api",
                    resilience=res,
                    operation=lambda: call_llm(spec),
                    result_failure=_llm_failure,
                    fallback=fallback,
                    retry_attempts=0,
                )
                resp = outcome.value
            except Exception as exc:
                e2e = time.monotonic() - start
                failure_result = getattr(exc, "result", None)
                record_error(
                    scenario_name,
                    t.provider,
                    str(exc),
                    e2e,
                    timed_out=getattr(failure_result, "timed_out", False),
                    retries=getattr(failure_result, "retries", 0),
                )
                log_request_event(scenario_name, e2e * 1000, exception=str(exc))
                span.set_attribute("error", True)
                return LayerResult(error=str(exc), row=row, duration_s=e2e)
            e2e = time.monotonic() - start
            span.set_attribute("llm_perf.degraded", outcome.degraded)
            err = (resp.raw or {}).get("error") if isinstance(resp.raw, dict) else None
            if err:
                record_error(
                    scenario_name,
                    t.provider,
                    str(err),
                    e2e,
                    timed_out=resp.timed_out,
                    retries=resp.retries,
                )
                log_request_event(scenario_name, e2e * 1000, exception=str(err))
                span.set_attribute("error", True)
                from ..security import redact_secrets

                span.set_attribute("llm_perf.error", redact_secrets(str(err))[:500])
                return LayerResult(error=str(err), response=resp, row=row, duration_s=e2e)
            record_response(scenario_name, resp, e2e, t.provider, model=model)
            # per-request breakdown for streaming: ttft + decode + queue shares
            from ..metrics import record_breakdown

            parts = {}
            if resp.ttft_s is not None:
                parts["ttft"] = resp.ttft_s * 1000.0
            if resp.generation_s is not None:
                parts["generation"] = resp.generation_s * 1000.0
            if resp.queue_wait_s is not None:
                parts["queue_wait"] = resp.queue_wait_s * 1000.0
            if parts:
                record_breakdown(scenario_name, e2e * 1000.0, parts)
            log_request_event(scenario_name, e2e * 1000)
            span.set_attribute("gen_ai.usage.input_tokens", resp.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.completion_tokens)
            if resp.finish_reason:
                span.set_attribute("gen_ai.response.finish_reasons", [resp.finish_reason])
            return LayerResult(
                output_text=resp.text,
                response=resp,
                row=row,
                duration_s=e2e,
                degraded=outcome.degraded,
            )
