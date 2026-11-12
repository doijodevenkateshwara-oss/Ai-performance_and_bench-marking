"""Locust task: execute one request through a layer + quality evaluation."""
from __future__ import annotations

import random
from typing import Optional

from .config import PerfConfig, ScenarioConfig
from .layers import LayerFactory
from .quality import JudgeSampler, evaluate_quality
from .quality.scorer import QualityResult


class RequestContext:
    """Holds per-scenario runtime state (factory, judge samplers)."""

    def __init__(self, config: PerfConfig):
        self.config = config
        self.factory = LayerFactory(config)
        self._judge_samplers: dict[str, JudgeSampler] = {}
        self._layers: dict[str, object] = {}

    def judge_sampler(self, scenario: ScenarioConfig) -> Optional[JudgeSampler]:
        if not scenario.quality or not scenario.quality.judge:
            return None
        key = scenario.name
        if key not in self._judge_samplers:
            self._judge_samplers[key] = JudgeSampler(scenario.quality.sample_rate)
        return self._judge_samplers[key]

    def run_scenario(self, scenario: ScenarioConfig, rng: random.Random) -> QualityResult:
        layer = self.factory.layer(scenario.target)
        prompt, row = self.factory.prompt_and_row(layer, rng)
        params = dict(scenario.params)
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
            from .metrics import record_quality
            record_quality(scenario.name, qres.score, qres.passed,
                           context_recall=qres.context_recall)
        return qres
