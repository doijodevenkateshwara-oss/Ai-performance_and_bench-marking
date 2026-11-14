"""Layer 2a: declarative pipeline (RAG/agent) targets.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import time

from ..config import PipelineTarget
from ..metrics import record_error, record_response, request_span
from ..pipelines import run_pipeline
from ..prompts import PromptPicker
from ..providers.base import NormalizedResponse
from .base import LayerResult


class PipelineLayer:
    """Declarative pipeline layer wrapping run_pipeline()."""

    def __init__(self, target: PipelineTarget, picker: PromptPicker, pricing=None):
        """PipelineLayer(target, picker, pricing)."""
        self.target = target
        self.picker = picker
        self.pricing = pricing

    def run(
        self, prompt: str, row: dict | None, scenario_params: dict, scenario_name: str = "default"
    ) -> LayerResult:
        """Run the stage pipeline, record response + agent-eval metrics."""
        t = self.target
        question = prompt or (row or {}).get("prompt") or (row or {}).get("question") or ""
        # build a synthetic NormalizedResponse-like usage record from ctx afterwards
        with request_span(scenario_name, t.name, "pipeline") as span:
            start = time.monotonic()
            try:
                params = dict(scenario_params)
                params["_target_name"] = t.name
                params["_target_resilience"] = t.resilience
                ctx = run_pipeline(t.pipeline, question, row, scenario_name, params, pricing=self.pricing)
            except Exception as e:
                e2e = time.monotonic() - start
                failure_result = getattr(e, "result", None)
                record_error(
                    scenario_name,
                    "pipeline",
                    str(e),
                    e2e,
                    timed_out=getattr(failure_result, "timed_out", False),
                    retries=getattr(failure_result, "retries", 0),
                )
                from ..metrics import log_request_event

                log_request_event(scenario_name, e2e * 1000, exception=str(e))
                span.set_attribute("error", True)
                return LayerResult(error=str(e), row=row, duration_s=e2e)
            e2e = time.monotonic() - start
            # per-request latency breakdown + IR metrics (composition is
            # aggregated per request — never by subtracting percentiles)
            from ..metrics import record_breakdown, record_ir_metrics

            parts = {k: v * 1000.0 for k, v in (ctx.stage_durations or {}).items()}
            if parts:
                record_breakdown(scenario_name, e2e * 1000.0, parts)
            if ctx.ir_metrics:
                record_ir_metrics(scenario_name, ctx.ir_metrics)
            # agent-stage evaluation metrics (success/tool/loop counters)
            if ctx.agent_eval:
                from ..metrics import record_agent_eval

                record_agent_eval(scenario_name, ctx.agent_eval)
            resp = NormalizedResponse(
                text=ctx.output_text,
                prompt_tokens=ctx.usage.get("prompt_tokens", 0),
                completion_tokens=ctx.usage.get("completion_tokens", 0),
            )
            resp.total_tokens = resp.tokens
            # record via the standard path so reports/thresholds work
            record_response(
                scenario_name,
                resp,
                e2e,
                "pipeline",
                model=(t.pipeline[-1].llm or {}).get("model") if t.pipeline else None,
            )
            from ..metrics import log_request_event

            log_request_event(scenario_name, e2e * 1000)
            span.set_attribute("gen_ai.usage.input_tokens", resp.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.completion_tokens)
            return LayerResult(
                output_text=ctx.output_text,
                response=resp,
                row=row,
                retrieved_texts=ctx.context_texts,
                duration_s=e2e,
                degraded=ctx.degraded,
                agent_success=ctx.agent_eval.get("success"),
                cost_attributed=True,
            )
