"""Bridge between OTel instruments and Locust custom stats.

Locust custom stats entries give us per-scenario aggregation for reports and
threshold evaluation; OTel instruments feed traces/metrics export. Both are
recorded from the same call sites.


Author: Venkateshwara Doijode
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
    """Return the cached OTel tracer (created lazily)."""
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
    """Return the current Locust RequestStats or None outside a run."""
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
        avg = sum(s) / n
        var = sum((x - avg) ** 2 for x in s) / n if n > 1 else 0.0
        out[(scenario, metric)] = {
            "count": n,
            "sum": sum(s),
            "avg": avg,
            "std": var**0.5,
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
    metric_attrs = {
        "scenario": scenario,
        "provider": provider,
        "target": target,
        "stream": stream,
    }
    if model:
        metric_attrs["model"] = model
    with tracer.start_as_current_span("llm_perf.request", attributes=span_attrs) as span:
        try:
            yield span
        finally:
            # single recorder for request duration — covers success and error paths
            elapsed = time.monotonic() - start
            span.set_attribute("llm_perf.duration_s", elapsed)
            inst["request_duration"].record(elapsed, metric_attrs)


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
    # request_duration is recorded once by request_span — do not duplicate here
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
    # output tok/s over the generation window when known, else e2e - ttft
    gen_s = resp.generation_s
    if gen_s is None and resp.ttft_s is not None:
        gen_s = max(e2e_s - resp.ttft_s, 0.001)
    gen_s = gen_s if gen_s else max(e2e_s, 0.001)
    tps = resp.completion_tokens / gen_s if resp.completion_tokens else 0.0
    itps = resp.prompt_tokens / max(e2e_s, 0.001) if resp.prompt_tokens else 0.0
    if resp.prompt_tokens or resp.completion_tokens:
        if resp.prompt_tokens:
            inst["tokens"].add(resp.prompt_tokens, {**attrs, "kind": "input"})
        if resp.completion_tokens:
            inst["tokens"].add(resp.completion_tokens, {**attrs, "kind": "output"})
        if tps:
            with suppress(AttributeError):
                inst["tokens_per_second"].record(tps, attrs)
        if itps:
            with suppress(AttributeError):
                inst["input_tokens_per_second"].record(itps, attrs)
    if resp.retries:
        inst["retries"].add(resp.retries, attrs)
    if resp.timed_out:
        inst["timeouts"].add(1, attrs)

    _log_custom(
        _locust_stats(),
        scenario,
        ttft_s=resp.ttft_s,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        tokens_per_sec=tps,
        input_tokens_per_sec=itps,
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
    """Record a failed request: OTel counter, Locust error entry, redacted stats."""
    inst = instruments()
    inst["request_total"].add(1, {"scenario": scenario, "provider": provider, "result": "error"})
    # request_duration is recorded once by request_span — do not duplicate here
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
        from .security import redact_secrets

        with suppress(Exception):
            stats.log_error("LLM_ERROR", scenario, redact_secrets(error)[:500])


def record_quality(
    scenario: str,
    score: float | None,
    passed: bool | None,
    check: str | None = None,
    context_recall: float | None = None,
):
    """Record quality score / pass / context-recall samples."""
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
    """Record one pipeline-stage or agent-step duration sample."""
    inst = instruments()
    inst["stage_duration"].record(duration_s, {"scenario": scenario, "stage": stage, **attrs})
    record_stage_sample(scenario, stage, duration_s)


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
        from .security import redact_secrets

        _CURRENT_ENV.events.request.fire(
            request_type=method,
            name=name,
            response_time=response_time_ms,
            response_length=response_length,
            response=None,
            context={},
            exception=redact_secrets(exception) if exception else exception,
        )


_CURRENT_USERS = 0
_PEAK_USERS = 0


def record_users(delta: int):
    """Adjust active-user count and the peak-users gauge."""
    global _CURRENT_USERS, _PEAK_USERS
    _CURRENT_USERS = max(_CURRENT_USERS + delta, 0)
    _PEAK_USERS = max(_PEAK_USERS, _CURRENT_USERS)
    instruments()["users"].add(delta)


def peak_users() -> int:
    """Return the peak concurrent virtual users seen this run."""
    return _PEAK_USERS


def record_threshold(rule: str, passed: bool):
    """Publish threshold pass/fail as a gauge (0/1) for live alerting."""
    g = instruments()["threshold_pass"]
    with suppress(AttributeError):
        g.set(1.0 if passed else 0.0, {"rule": rule})


# ---------------------------------------------------------------------------
# Stage-duration aggregation (for threshold `stages` rules + reports)
# ---------------------------------------------------------------------------

_STAGE_LOCK = threading.Lock()
_STAGE_SAMPLES: dict[tuple[str, str], list[float]] = defaultdict(list)
_STAGE_CAP = 20000


def record_stage_sample(scenario: str, stage: str, duration_s: float):
    """Store an exact stage-duration sample for reports/thresholds."""
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


# ---------------------------------------------------------------------------
# Per-request latency breakdown — statistically valid composition.
#
# Percentiles are NOT additive: p95(e2e) != sum of stage p95s because each
# percentile may come from a different request. Instead we store per-request
# (e2e, {segment: ms}) tuples and aggregate *within* each request first —
# shares and residuals are computed per request, then percentiled.
# ---------------------------------------------------------------------------

_BREAKDOWN_LOCK = threading.Lock()
_BREAKDOWN: dict[str, list[dict]] = defaultdict(list)
_BREAKDOWN_CAP = 20000


def record_breakdown(scenario: str, e2e_ms: float, parts: dict[str, float]):
    """Store one request's segment durations (ms) alongside its e2e (ms)."""
    if e2e_ms is None or e2e_ms <= 0 or not parts:
        return
    clean = {k: float(v) for k, v in parts.items() if v is not None and v >= 0}
    if not clean:
        return
    with _BREAKDOWN_LOCK:
        lst = _BREAKDOWN[scenario]
        if len(lst) < _BREAKDOWN_CAP:
            lst.append({"e2e_ms": float(e2e_ms), "parts": clean})


def _pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of a sorted list (q in 0..1)."""
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(int(len(sorted_vals) * q), len(sorted_vals) - 1)]


def breakdown_summary() -> dict[str, dict]:
    """Aggregate per-request breakdowns into composition stats per scenario.

    Returns {scenario: {"requests": n, "segments": {name: stats},
    "unattributed": stats}} where stats = {p50_ms, p95_ms, median_share_pct}.
    Shares are per-request (segment/e2e) medians — they sum to <=100% with the
    unattributed remainder covering queueing/overhead between segments.
    """
    with _BREAKDOWN_LOCK:
        items = {k: list(v) for k, v in _BREAKDOWN.items()}
    out: dict[str, dict] = {}
    for scenario, reqs in items.items():
        if not reqs:
            continue
        seg_names = sorted({seg for r in reqs for seg in r["parts"]})
        seg_abs: dict[str, list[float]] = {s: [] for s in seg_names}
        seg_share: dict[str, list[float]] = {s: [] for s in seg_names}
        un_abs: list[float] = []
        un_share: list[float] = []
        for r in reqs:
            e2e = r["e2e_ms"]
            total = sum(r["parts"].values())
            residual = max(e2e - total, 0.0)
            un_abs.append(residual)
            un_share.append(residual / e2e if e2e > 0 else 0.0)
            for s in seg_names:
                v = r["parts"].get(s)
                if v is None:
                    continue
                seg_abs[s].append(v)
                seg_share[s].append(v / e2e if e2e > 0 else 0.0)

        def _stats(abs_vals: list[float], share_vals: list[float]) -> dict:
            """p50/p95 absolute ms + median share for one segment."""
            a = sorted(abs_vals)
            sh = sorted(share_vals)
            return {
                "p50_ms": _pct(a, 0.50),
                "p95_ms": _pct(a, 0.95),
                "median_share_pct": round(_pct(sh, 0.50) * 100.0, 2) if sh else None,
            }

        out[scenario] = {
            "requests": len(reqs),
            "segments": {s: _stats(seg_abs[s], seg_share[s]) for s in seg_names},
            "unattributed": _stats(un_abs, un_share),
        }
    return out


# ---------------------------------------------------------------------------
# Cost by call-site stage (generate / rerank / agent / judge / ...)
# ---------------------------------------------------------------------------

_COST_STAGE_LOCK = threading.Lock()
_COST_STAGE: dict[tuple[str, str], float] = defaultdict(float)


def record_cost(scenario: str, cost_usd: float, stage: str | None = None):
    """Accumulate estimated USD cost, optionally tagged by call-site stage.

    OTel counter carries a `stage` attribute ("response" for the scenario-level
    rollup recorded by the task layer, stage names like generate/rerank/agent/
    judge for pipeline call sites). _COST_STAGE feeds report cost_breakdown.
    """
    if not cost_usd:
        return
    attrs = {"scenario": scenario}
    if stage:
        attrs["stage"] = stage
    instruments()["cost"].add(cost_usd, attrs)
    if stage:
        with _COST_STAGE_LOCK:
            _COST_STAGE[(scenario, stage)] += cost_usd


def cost_stage_snapshot() -> dict[str, dict[str, float]]:
    """{scenario: {stage: usd}} accumulated cost by call site."""
    with _COST_STAGE_LOCK:
        items = dict(_COST_STAGE)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (scenario, stage), usd in items.items():
        out[scenario][stage] = round(usd, 6)
    return dict(out)


# ---------------------------------------------------------------------------
# Information-retrieval metrics (Recall@K / Precision@K / MRR / NDCG)
# ---------------------------------------------------------------------------

_IR_METRICS = ("recall_at_k", "precision_at_k", "mrr", "ndcg")
_IR_LOCUST = (
    ("recall_at_k", "RECALL_AT_K"),
    ("precision_at_k", "PRECISION_AT_K"),
    ("mrr", "MRR"),
    ("ndcg", "NDCG"),
)


def record_ir_metrics(scenario: str, metrics: dict[str, float | None]):
    """Record per-request IR metrics as exact samples + Locust custom stats."""
    stats = _locust_stats()
    for key in _IR_METRICS:
        v = metrics.get(key)
        if v is None:
            continue
        record_metric(scenario, key, v)
        entry = dict(_IR_LOCUST).get(key)
        if stats is not None and entry:
            with suppress(Exception):
                stats.get(scenario, entry).log(v * 1000.0, 0)
