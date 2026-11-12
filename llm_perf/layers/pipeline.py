"""Layer 2a: declarative pipeline (RAG/agent) targets."""
from __future__ import annotations

import time
from typing import Optional

from ..config import PipelineTarget
from ..metrics import (instruments, record_error, record_response, request_span,
                       record_cost)
from ..pipelines import run_pipeline
from ..prompts import PromptPicker
from ..providers.base import NormalizedResponse
from .base import LayerResult


class PipelineLayer:
    def __init__(self, target: PipelineTarget, picker: PromptPicker):
        self.target = target
        self.picker = picker

    def run(self, prompt: str, row: Optional[dict], scenario_params: dict,
            scenario_name: str = "default") -> LayerResult:
        t = self.target
        question = prompt or (row or {}).get("prompt") or (row or {}).get("question") or ""
        # build a synthetic NormalizedResponse-like usage record from ctx afterwards
        with request_span(scenario_name, t.name, "pipeline") as span:
            start = time.monotonic()
            try:
                ctx = run_pipeline(t.pipeline, question, row, scenario_name,
                                   scenario_params)
            except Exception as e:
                e2e = time.monotonic() - start
                record_error(scenario_name, "pipeline", str(e), e2e)
                span.set_attribute("error", True)
                return LayerResult(error=str(e), row=row,
                                   stage_metrics={"e2e_s": e2e})
            e2e = time.monotonic() - start
            stage_metrics = {"e2e_s": e2e}
            # generate stage timing recorded via record_stage; e2e covers all
            resp = NormalizedResponse(
                text=ctx.output_text,
                prompt_tokens=ctx.usage.get("prompt_tokens", 0),
                completion_tokens=ctx.usage.get("completion_tokens", 0),
            )
            resp.total_tokens = resp.tokens
            # record via the standard path so reports/thresholds work
            record_response(scenario_name, resp, e2e, "pipeline",
                            model=(t.pipeline[-1].llm or {}).get("model")
                            if t.pipeline else None)
            span.set_attribute("gen_ai.usage.input_tokens", resp.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.completion_tokens)
            return LayerResult(output_text=ctx.output_text, response=resp, row=row,
                               stage_metrics=stage_metrics,
                               retrieved_texts=ctx.context_texts)
