"""Bridge between OTel instruments and Locust custom stats.

Locust custom stats entries give us per-scenario aggregation for reports and
threshold evaluation; OTel instruments feed traces/metrics export. Both are
recorded from the same call sites.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from llm_perf.otel.genai_semconv import instruments
from llm_perf.otel.setup import get_tracer
from llm_perf.providers.base import NormalizedResponse

_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is None:
        _tracer = get_tracer()
    return _tracer


def _locust_stats():
    try:
        from locust.env import get_environment
        return get_environment().stats
    except Exception:
        return None


def _log_custom(stats, name: str, ttft_s: Optional[float] = None,
                prompt_tokens: int = 0, completion_tokens: int = 0,
                tokens_per_sec: float = 0.0, quality_score: Optional[float] = None,
                context_recall: Optional[float] = None):
    """Record into Locust custom stats entries (created lazily)."""
    if stats is None:
        return
    try:
        if ttft_s is not None:
            e = stats.get(name, "TTFT", method="TTFT")
            e.log(ttft_s * 1000, name=name)
        if prompt_tokens or completion_tokens:
            e = stats.get(name, "TOKENS", method="TOKENS")
            e.log(prompt_tokens + completion_tokens, name=name)
            e2 = stats.get(name, "PROMPT_TOKENS", method="PROMPT_TOKENS")
            e2.log(prompt_tokens, name=name)
            e3 = stats.get(name, "COMPLETION_TOKENS", method="COMPLETION_TOKENS")
            e3.log(completion_tokens, name=name)
        if tokens_per_sec:
            e = stats.get(name, "TOKENS_PER_SEC", method="TOKENS_PER_SEC")
            e.log(tokens_per_sec, name=name)
        if quality_score is not None:
            e = stats.get(name, "QUALITY_SCORE", method="QUALITY_SCORE")
            e.log(quality_score * 1000, name=name)  # store as 0..1000
        if context_recall is not None:
            e = stats.get(name, "CONTEXT_RECALL", method="CONTEXT_RECALL")
            e.log(context_recall * 1000, name=name)
    except Exception:
        pass


@contextmanager
def request_span(scenario: str, target: str, provider: str,
                 model: Optional[str] = None, stream: bool = False,
                 use_genai_semconv: bool = True):
    """Root span + timing context for one virtual-user request."""
    tracer = _get_tracer()
    inst = instruments()
    start = time.monotonic()
    attrs = {
        "llm_perf.scenario": scenario,
        "llm_perf.target": target,
        "llm_perf.provider": provider,
        "llm_perf.stream": stream,
    }
    if model:
        attrs["gen_ai.request.model"] = model
    span_attrs = dict(attrs)
    if use_genai_semconv:
        span_attrs.setdefault("gen_ai.operation.name", "chat")
        span_attrs.setdefault("gen_ai.system", provider)
    with tracer.start_as_current_span("llm_perf.request", attributes=span_attrs) as span:
        yield span
        elapsed = time.monotonic() - start
        span.set_attribute("llm_perf.duration_s", elapsed)
        inst["request_duration"].record(elapsed, attrs)


def stage_span(name: str, scenario: str, **extra_attrs):
    """Child span for a pipeline stage / tool call."""
    tracer = _get_tracer()
    attrs = {"llm_perf.scenario": scenario, "llm_perf.stage": name, **extra_attrs}
    return tracer.start_as_current_span(f"llm_perf.stage.{name}", attributes=attrs)


def record_response(scenario: str, resp: NormalizedResponse,
                    e2e_s: float, provider: str, model: Optional[str] = None,
                    use_genai_semconv: bool = True):
    """Record metrics for a completed LLM call (OTel + Locust stats)."""
    inst = instruments()
    attrs = {"scenario": scenario, "provider": provider}
    if model:
        attrs["model"] = model

    inst["request_total"].add(1, {**attrs, "result": "success"})
    inst["request_duration"].record(e2e_s, attrs)
    if resp.ttft_s is not None:
        inst["ttft"].record(resp.ttft_s, attrs)
    if resp.prompt_tokens or resp.completion_tokens:
        if resp.prompt_tokens:
            inst["tokens"].add(resp.prompt_tokens, {**attrs, "kind": "input"})
        if resp.completion_tokens:
            inst["tokens"].add(resp.completion_tokens, {**attrs, "kind": "output"})
        gen_ms = max(e2e_s - (resp.ttft_s or 0), 0.001)
        tps = resp.completion_tokens / gen_ms if resp.completion_tokens else 0.0
        if tps:
            inst["tokens_per_second"].record(tps, attrs)

    _log_custom(_locust_stats(), scenario,
                ttft_s=resp.ttft_s, prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens)


def record_error(scenario: str, provider: str, error: str, e2e_s: float = 0.0):
    inst = instruments()
    inst["request_total"].add(1, {"scenario": scenario, "provider": provider,
                                  "result": "error"})
    if e2e_s:
        inst["request_duration"].record(e2e_s,
                                       {"scenario": scenario, "provider": provider})
    stats = _locust_stats()
    if stats is not None:
        try:
            stats.log_error(scenario, error[:500])
        except Exception:
            pass


def record_quality(scenario: str, score: Optional[float], passed: Optional[bool],
                   check: Optional[str] = None, context_recall: Optional[float] = None):
    inst = instruments()
    if score is not None:
        inst["quality_score"].record(score, {"scenario": scenario})
        _log_custom(_locust_stats(), scenario, quality_score=score)
    if passed is not None:
        inst["quality_pass"].add(1, {"scenario": scenario,
                                     "result": "pass" if passed else "fail",
                                     "check": check or "overall"})
    if context_recall is not None:
        inst["context_recall"].record(context_recall, {"scenario": scenario})
        _log_custom(_locust_stats(), scenario, context_recall=context_recall)


def record_stage(scenario: str, stage: str, duration_s: float, **attrs):
    inst = instruments()
    inst["stage_duration"].record(duration_s,
                                 {"scenario": scenario, "stage": stage, **attrs})


def record_cost(scenario: str, cost_usd: float):
    if cost_usd:
        instruments()["cost"].add(cost_usd, {"scenario": scenario})


def record_users(delta: int):
    instruments()["users"].add(delta)
