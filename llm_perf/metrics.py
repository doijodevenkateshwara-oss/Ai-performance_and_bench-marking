"""Bridge between OTel instruments and Locust custom stats.

Locust custom stats entries give us per-scenario aggregation for reports and
threshold evaluation; OTel instruments feed traces/metrics export. Both are
recorded from the same call sites.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager, suppress

from llm_perf.otel.genai_semconv import instruments
from llm_perf.otel.setup import get_tracer
from llm_perf.providers.base import NormalizedResponse

_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is None:
        _tracer = get_tracer()
    return _tracer


_CURRENT_ENV = None


def set_environment(env) -> None:
    """Called by generated TaskSets so metric helpers can reach Locust stats."""
    global _CURRENT_ENV
    _CURRENT_ENV = env


def _locust_stats():
    return getattr(_CURRENT_ENV, "stats", None) if _CURRENT_ENV is not None else None


def _log_custom(
    stats,
    name: str,
    ttft_s: float | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    tokens_per_sec: float = 0.0,
    quality_score: float | None = None,
    context_recall: float | None = None,
):
    """Record into Locust custom stats entries (created lazily)."""
    if stats is None:
        return
    try:
        if ttft_s is not None:
            e = stats.get(name, "TTFT")
            e.log(int(ttft_s * 1000), 0)
        if prompt_tokens or completion_tokens:
            e = stats.get(name, "TOKENS")
            e.log(prompt_tokens + completion_tokens, 0)
            e2 = stats.get(name, "PROMPT_TOKENS")
            e2.log(prompt_tokens, 0)
            e3 = stats.get(name, "COMPLETION_TOKENS")
            e3.log(completion_tokens, 0)
        if tokens_per_sec:
            e = stats.get(name, "TOKENS_PER_SEC")
            e.log(int(tokens_per_sec * 100), 0)
        if quality_score is not None:
            e = stats.get(name, "QUALITY_SCORE")
            e.log(int(quality_score * 1000), 0)  # store as 0..1000
        if context_recall is not None:
            e = stats.get(name, "CONTEXT_RECALL")
            e.log(int(context_recall * 1000), 0)
    except Exception:
        pass


@contextmanager
def request_span(
    scenario: str,
    target: str,
    provider: str,
    model: str | None = None,
    stream: bool = False,
    use_genai_semconv: bool = True,
):
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


def record_response(
    scenario: str,
    resp: NormalizedResponse,
    e2e_s: float,
    provider: str,
    model: str | None = None,
    use_genai_semconv: bool = True,
):
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
            with suppress(AttributeError):
                inst["tokens_per_second"].set(tps, attrs)

    _log_custom(
        _locust_stats(),
        scenario,
        ttft_s=resp.ttft_s,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
    )


def record_error(scenario: str, provider: str, error: str, e2e_s: float = 0.0):
    inst = instruments()
    inst["request_total"].add(1, {"scenario": scenario, "provider": provider, "result": "error"})
    if e2e_s:
        inst["request_duration"].record(e2e_s, {"scenario": scenario, "provider": provider})
    stats = _locust_stats()
    if stats is not None:
        with suppress(Exception):
            stats.log_error("LLM_ERROR", scenario, error[:500])


def record_quality(
    scenario: str,
    score: float | None,
    passed: bool | None,
    check: str | None = None,
    context_recall: float | None = None,
):
    inst = instruments()
    if score is not None:
        inst["quality_score"].record(score, {"scenario": scenario})
        _log_custom(_locust_stats(), scenario, quality_score=score)
    if passed is not None:
        inst["quality_pass"].add(
            1, {"scenario": scenario, "result": "pass" if passed else "fail", "check": check or "overall"}
        )
    if context_recall is not None:
        inst["context_recall"].record(context_recall, {"scenario": scenario})
        _log_custom(_locust_stats(), scenario, context_recall=context_recall)


def record_stage(scenario: str, stage: str, duration_s: float, **attrs):
    inst = instruments()
    inst["stage_duration"].record(duration_s, {"scenario": scenario, "stage": stage, **attrs})
    record_stage_sample(scenario, stage, duration_s)


def record_cost(scenario: str, cost_usd: float):
    if cost_usd:
        instruments()["cost"].add(cost_usd, {"scenario": scenario})


def log_request_event(
    name: str,
    response_time_ms: float,
    exception: str | None = None,
    method: str = "POST",
    response_length: int = 0,
):
    """Report one completed request into Locust's stats (RPS/lat/errors).

    Fires the standard `request` event; the runner's listener records it into
    RequestStats so scenarios appear in Locust stats/UI/reports.
    """
    if _CURRENT_ENV is None:
        return
    with suppress(Exception):
        _CURRENT_ENV.events.request.fire(
            request_type=method,
            name=name,
            response_time=response_time_ms,
            response_length=response_length,
            response=None,
            context={},
            exception=exception,
        )


def record_users(delta: int):
    instruments()["users"].add(delta)


def record_threshold(rule: str, passed: bool):
    """Publish threshold pass/fail as a gauge (0/1) for live alerting."""
    g = instruments()["threshold_pass"]
    with suppress(AttributeError):
        g.set(1.0 if passed else 0.0, {"rule": rule})
    with suppress(Exception):
        g.record(1.0 if passed else 0.0, {"rule": rule})


# ---------------------------------------------------------------------------
# Stage-duration aggregation (for threshold `stages` rules + reports)
# ---------------------------------------------------------------------------

_STAGE_LOCK = threading.Lock()
_STAGE_SAMPLES: dict[tuple[str, str], list[float]] = defaultdict(list)
_STAGE_CAP = 20000


def record_stage_sample(scenario: str, stage: str, duration_s: float):
    with _STAGE_LOCK:
        lst = _STAGE_SAMPLES[(scenario, stage)]
        if len(lst) < _STAGE_CAP:
            lst.append(duration_s)


def stage_stats_snapshot() -> dict[tuple[str, str], dict]:
    """Aggregate stage durations into {key: {p50_ms,p95_ms,avg_ms,max_ms,count}}."""
    out = {}
    with _STAGE_LOCK:
        items = {k: list(v) for k, v in _STAGE_SAMPLES.items()}
    for key, samples in items.items():
        if not samples:
            continue
        s = sorted(samples)
        n = len(s)
        out[key] = {
            "count": n,
            "avg_ms": sum(s) / n * 1000.0,
            "p50_ms": s[int(n * 0.50)] * 1000.0,
            "p95_ms": s[min(int(n * 0.95), n - 1)] * 1000.0,
            "max_ms": s[-1] * 1000.0,
        }
    return out


def estimate_cost(prompt_tokens: int, completion_tokens: int, pricing) -> float:
    """pricing: PricingConfig | None -> USD."""
    if pricing is None:
        return 0.0
    return (prompt_tokens / 1000.0) * pricing.input_per_1k + (
        completion_tokens / 1000.0
    ) * pricing.output_per_1k
