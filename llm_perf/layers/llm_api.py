"""Layer 1: direct LLM API calls via provider adapters.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import os
import time

from ..config import LlmApiTarget
from ..metrics import log_request_event, record_error, record_response, request_span
from ..prompts import PromptPicker
from ..providers import RequestSpec, call_llm
from .base import LayerResult


class LlmApiLayer:
    """Direct vendor-API layer: builds a RequestSpec and calls call_llm."""

    def __init__(self, target: LlmApiTarget, picker: PromptPicker):
        """LlmApiLayer(target, picker)."""
        self.target = target
        self.picker = picker

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
                "grpc": t.grpc.model_dump() if t.grpc else {},
                "request_template": t.request_template,
                "response_mapping": t.response_mapping,
                "headers": t.tags.get("headers", {}),
                "retry": {"max_retries": t.retries, "backoff_s": t.retry_backoff_s},
            },
        )
        model = params.get("model") or params.get("model_id")
        with request_span(scenario_name, t.name, t.provider, model=model, stream=spec.stream) as span:
            start = time.monotonic()
            resp = call_llm(spec)
            e2e = time.monotonic() - start
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
                return LayerResult(error=str(err), response=resp, row=row)
            record_response(scenario_name, resp, e2e, t.provider, model=model)
            log_request_event(scenario_name, e2e * 1000)
            span.set_attribute("gen_ai.usage.input_tokens", resp.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.completion_tokens)
            if resp.finish_reason:
                span.set_attribute("gen_ai.response.finish_reasons", [resp.finish_reason])
            return LayerResult(
                output_text=resp.text,
                response=resp,
                row=row,
            )
