"""Locust task: execute one request through a layer + quality evaluation.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import random
import time

from .config import PerfConfig, ScenarioConfig
from .layers import LayerFactory
from .quality import JudgeSampler, evaluate_quality
from .quality.scorer import QualityResult


class RequestContext:
    """Holds per-scenario runtime state (factory, judge samplers, cost)."""

    def __init__(self, config: PerfConfig):
        """RequestContext(config): shared factory, judge samplers, cost totals."""
        self.config = config
        self.factory = LayerFactory(config)
        self._judge_samplers: dict[str, JudgeSampler] = {}
        self.cost_by_scenario: dict[str, float] = {}
        self.pricing = getattr(config.report, "pricing", None)

    def judge_sampler(self, scenario: ScenarioConfig) -> JudgeSampler | None:
        """Return (creating if needed) the JudgeSampler for a scenario."""
        if not scenario.quality or not scenario.quality.judge:
            return None
        key = scenario.name
        if key not in self._judge_samplers:
            self._judge_samplers[key] = JudgeSampler(scenario.quality.sample_rate)
        return self._judge_samplers[key]

    def run_scenario(self, scenario: ScenarioConfig, rng: random.Random) -> QualityResult:
        """Run one scenario end-to-end: layer call + quality + cost accounting."""
        layer = self.factory.layer(scenario.target)
        prompt, row = self.factory.prompt_and_row(layer, rng)
        params = dict(scenario.params)
        params["_task_start"] = time.monotonic()  # queue-wait measurement origin
        if scenario.stream is not None:
            params["_stream"] = scenario.stream
        result = layer.run(prompt, row, params, scenario_name=scenario.name)

        qres = QualityResult()
        if scenario.quality and not result.error:
            expected_ctx = None
            if row and isinstance(row, dict):
                expected_ctx = row.get("expected_context") or row.get("context")
            qres = evaluate_quality(
                scenario.quality,
                question=prompt,
                response=result.output_text,
                row=row,
                retrieved_texts=result.retrieved_texts or None,
                expected_context=expected_ctx,
                judge_sampler=self.judge_sampler(scenario),
            )
            from .metrics import record_metric, record_quality

            record_quality(scenario.name, qres.score, qres.passed, context_recall=qres.context_recall)
            record_metric(scenario.name, "quality_eligible", 1.0)
            record_metric(scenario.name, "quality_evaluated", 1.0)
            if qres.judged:
                record_metric(scenario.name, "quality_judged", 1.0)
            # judge calls are real LLM spend outside the response token counts
            if qres.judge_usage and self.pricing is not None:
                from .metrics import estimate_cost, record_cost

                jcost = estimate_cost(
                    qres.judge_usage.get("prompt_tokens", 0),
                    qres.judge_usage.get("completion_tokens", 0),
                    self.pricing,
                )
                if jcost:
                    record_cost(scenario.name, jcost, stage="judge")
                    self.cost_by_scenario[scenario.name] = (
                        self.cost_by_scenario.get(scenario.name, 0.0) + jcost
                    )

        # cost accounting (skip quality-only/judge calls — they aren't in result)
        # pipeline layers already attribute cost per stage (generate/rerank/agent);
        # only unattributed layers (llm_api etc.) record the rollup here.
        resp = result.response
        if resp is not None and (resp.prompt_tokens or resp.completion_tokens):
            from .metrics import estimate_cost, record_cost

            cost = estimate_cost(resp.prompt_tokens, resp.completion_tokens, self.pricing)
            if cost:
                record_cost(scenario.name, cost, stage=None if result.cost_attributed else "generation")
                self.cost_by_scenario[scenario.name] = self.cost_by_scenario.get(scenario.name, 0.0) + cost
        return qres
