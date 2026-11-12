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


# ---------------------------------------------------------------------------
# Exact custom-metric samples — authoritative values for reports/thresholds.
# Locust StatsEntry buckets percentiles to ~2 significant digits and is built
# for request latencies; these samples keep full precision.
# ---------------------------------------------------------------------------

_CUSTOM_LOCK = threading.Lock()
_CUSTOM: dict[tuple[str, str], list[float]] = defaultdict(list)
_CUSTOM_CAP = 20000


def record_metric(scenario: str, metric: str, value: float):
    """Store one exact sample for (scenario, metric)."""
    if value is None:
        return
    with _CUSTOM_LOCK:
        lst = _CUSTOM[(scenario, metric)]
        if len(lst) < _CUSTOM_CAP:
            lst.append(float(value))


def custom_stats_snapshot() -> dict[tuple[str, str], dict]:
    """Aggregate samples into {(scenario, metric): {count,sum,avg,min,max,p50,p95}}."""
    out = {}
    with _CUSTOM_LOCK:
        items = {k: list(v) for k, v in _CUSTOM.items()}
    for (scenario, metric), samples in items.items():
        if not samples:
            continue
        s = sorted(samples)
        n = len(s)
        out[(scenario, metric)] = {
            "count": n,
            "sum": sum(s),
            "avg": sum(s) / n,
            "min": s[0],
            "max": s[-1],
            "p50": s[int(n * 0.50)],
            "p95": s[min(int(n * 0.95), n - 1)],
        }
    return out


def _log_custom(
    stats,
    name: str,
    ttft_s: float | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    tokens_per_sec: float = 0.0,
    input_tokens_per_sec: float = 0.0,
    tpot_s: float | None = None,
    generation_s: float | None = None,
    itl_avg_ms: float | None = None,
    itl_p95_ms: float | None = None,
    queue_wait_s: float | None = None,
    retries: int = 0,
    timed_out: bool = False,
    quality_score: float | None = None,
    context_recall: float | None = None,
):
    """Record into Locust custom stats entries (created lazily).

    Also stores exact samples via record_metric for report/threshold use —
    the Locust entries are scaled/bucketed approximations for the live UI.
    """
    # exact samples (always, even without a Locust env)
    exact = [
        ("ttft_ms", ttft_s * 1000.0 if ttft_s is not None else None),
        ("tokens_per_sec", tokens_per_sec or None),
        ("input_tokens_per_sec", input_tokens_per_sec or None),
        ("tpot_ms", tpot_s * 1000.0 if tpot_s is not None else None),
        ("gen_duration_ms", generation_s * 1000.0 if generation_s is not None else None),
        ("itl_avg_ms", itl_avg_ms),
        ("itl_p95_ms", itl_p95_ms),
        ("queue_wait_ms", queue_wait_s * 1000.0 if queue_wait_s is not None else None),
        ("retries", float(retries) if retries else None),
        ("timeouts", 1.0 if timed_out else None),
        ("quality_score", quality_score),
        ("context_recall", context_recall),
    ]
    for metric, value in exact:
        if value is not None:
            record_metric(name, metric, value)

    if stats is None:
        return
    # Locust custom-stat entries — live-UI approximations (scaled/bucketed);
    # reports and thresholds use the exact samples above.
    entries = [
        ("TTFT", ttft_s * 1000.0 if ttft_s is not None else None),
        ("TOKENS_PER_SEC", tokens_per_sec or None),
        ("INPUT_TOKENS_PER_SEC", input_tokens_per_sec or None),
        ("TPOT", tpot_s * 1000.0 if tpot_s is not None else None),
        ("GEN_DURATION", generation_s * 1000.0 if generation_s is not None else None),
        ("ITL_AVG", itl_avg_ms),
        ("ITL_P95", itl_p95_ms),
        ("QUEUE_WAIT", queue_wait_s * 1000.0 if queue_wait_s is not None else None),
        ("RETRIES", float(retries) if retries else None),
        ("TIMEOUTS", 1.0 if timed_out else None),
        ("QUALITY_SCORE", quality_score * 1000.0 if quality_score is not None else None),
        ("CONTEXT_RECALL", context_recall * 1000.0 if context_recall is not None else None),
    ]
    try:
        if prompt_tokens or completion_tokens:
            stats.get(name, "TOKENS").log(prompt_tokens + completion_tokens, 0)
            stats.get(name, "PROMPT_TOKENS").log(prompt_tokens, 0)
            stats.get(name, "COMPLETION_TOKENS").log(completion_tokens, 0)
        for label, value in entries:
            if value is not None:
                stats.get(name, label).log(value, 0)
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
    ss = resp.stream_stats or {}
    if resp.ttft_s is not None:
        inst["ttft"].record(resp.ttft_s, attrs)
    if resp.tpot_s is not None:
        with suppress(AttributeError):
            inst["tpot"].record(resp.tpot_s, attrs)
    if resp.generation_s is not None:
        with suppress(AttributeError):
            inst["generation_duration"].record(resp.generation_s, attrs)
    if ss.get("itl_avg_ms") is not None:
        with suppress(AttributeError):
            inst["inter_token_latency"].record(ss["itl_avg_ms"] / 1000.0, attrs)
    if resp.queue_wait_s is not None:
        with suppress(AttributeError):
            inst["queue_wait"].record(resp.queue_wait_s, attrs)
    if resp.prompt_tokens or resp.completion_tokens:
        if resp.prompt_tokens:
            inst["tokens"].add(resp.prompt_tokens, {**attrs, "kind": "input"})
        if resp.completion_tokens:
            inst["tokens"].add(resp.completion_tokens, {**attrs, "kind": "output"})
        # output tok/s over the generation window when known, else e2e - ttft
        gen_s = resp.generation_s
        if gen_s is None and resp.ttft_s is not None:
            gen_s = max(e2e_s - resp.ttft_s, 0.001)
        gen_s = gen_s if gen_s else max(e2e_s, 0.001)
        tps = resp.completion_tokens / gen_s if resp.completion_tokens else 0.0
        if tps:
            with suppress(AttributeError):
                inst["tokens_per_second"].set(tps, attrs)
        itps = resp.prompt_tokens / max(e2e_s, 0.001) if resp.prompt_tokens else 0.0
        if itps:
            with suppress(AttributeError):
                inst["input_tokens_per_second"].set(itps, attrs)

    _log_custom(
        _locust_stats(),
        scenario,
        ttft_s=resp.ttft_s,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        tokens_per_sec=(resp.completion_tokens / gen_s) if resp.completion_tokens else 0.0,
        input_tokens_per_sec=(resp.prompt_tokens / max(e2e_s, 0.001)) if resp.prompt_tokens else 0.0,
        tpot_s=resp.tpot_s,
        generation_s=resp.generation_s,
        itl_avg_ms=ss.get("itl_avg_ms"),
        itl_p95_ms=ss.get("itl_p95_ms"),
        queue_wait_s=resp.queue_wait_s,
        retries=resp.retries,
        timed_out=resp.timed_out,
    )


def record_error(
    scenario: str,
    provider: str,
    error: str,
    e2e_s: float = 0.0,
    timed_out: bool = False,
    retries: int = 0,
):
    inst = instruments()
    inst["request_total"].add(1, {"scenario": scenario, "provider": provider, "result": "error"})
    if e2e_s:
        inst["request_duration"].record(e2e_s, {"scenario": scenario, "provider": provider})
    stats = _locust_stats()
    if retries:
        record_metric(scenario, "retries", float(retries))
        with suppress(AttributeError):
            inst["retries"].add(retries, {"scenario": scenario, "provider": provider})
        if stats is not None:
            with suppress(Exception):
                stats.get(scenario, "RETRIES").log(retries, 0)
    if timed_out:
        record_metric(scenario, "timeouts", 1.0)
        with suppress(AttributeError):
            inst["timeouts"].add(1, {"scenario": scenario, "provider": provider})
        if stats is not None:
            with suppress(Exception):
                stats.get(scenario, "TIMEOUTS").log(1, 0)
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
        record_metric(scenario, "quality_pass", 1.0 if passed else 0.0)
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


def record_agent_eval(scenario: str, ev: dict):
    """Record agent-stage evaluation metrics (OTel + exact samples + Locust stats)."""
    if not ev:
        return
    inst = instruments()
    attrs = {"scenario": scenario}
    steps = ev.get("steps") or 0
    tool_calls = ev.get("tool_calls") or 0
    tool_errors = ev.get("tool_errors") or 0
    loops = ev.get("loops") or 0
    success = ev.get("success")
    terminated = ev.get("terminated") or "unknown"

    with suppress(AttributeError):
        inst["agent_steps"].record(steps, attrs)
        inst["agent_tool_calls"].add(tool_calls, attrs)
        inst["agent_tool_errors"].add(tool_errors, attrs)
        inst["agent_loops"].add(loops, attrs)
        if success is not None:
            inst["agent_task_success"].add(1, {**attrs, "result": "success" if success else "failure"})

    # exact samples for reports/thresholds
    record_metric(scenario, "agent_run", 1.0)
    record_metric(scenario, "agent_steps", float(steps))
    record_metric(scenario, "agent_tool_calls", float(tool_calls))
    record_metric(scenario, "agent_tool_errors", float(tool_errors))
    record_metric(scenario, "agent_loops", float(loops))
    record_metric(scenario, "agent_terminated_" + terminated, 1.0)
    if success is not None:
        record_metric(scenario, "agent_success", 1.0 if success else 0.0)

    stats = _locust_stats()
    if stats is not None:
        with suppress(Exception):
            stats.get(scenario, "AGENT_STEPS").log(steps, 0)
            stats.get(scenario, "AGENT_TOOL_CALLS").log(tool_calls, 0)
            stats.get(scenario, "AGENT_TOOL_ERRORS").log(tool_errors, 0)
            stats.get(scenario, "AGENT_LOOPS").log(loops, 0)
            if success is not None:
                stats.get(scenario, "AGENT_SUCCESS").log(1 if success else 0, 0)


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


_CURRENT_USERS = 0
_PEAK_USERS = 0


def record_users(delta: int):
    global _CURRENT_USERS, _PEAK_USERS
    _CURRENT_USERS = max(_CURRENT_USERS + delta, 0)
    _PEAK_USERS = max(_PEAK_USERS, _CURRENT_USERS)
    instruments()["users"].add(delta)


def peak_users() -> int:
    return _PEAK_USERS


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
