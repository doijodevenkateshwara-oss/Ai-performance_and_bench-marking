"""Resilience runtime and integration tests.

Author: Venkateshwara Doijode
"""

import time
from types import SimpleNamespace

import pytest
import requests

from llm_perf import metrics
from llm_perf.config import (
    AppEndpointTarget,
    LlmApiTarget,
    PerfConfig,
    PipelineStage,
    ResilienceConfig,
    ThresholdsConfig,
)
from llm_perf.layers.app_endpoint import AppEndpointLayer
from llm_perf.layers.llm_api import LlmApiLayer
from llm_perf.pipelines.engine import StageContext, run_stage_generate
from llm_perf.providers.base import NormalizedResponse
from llm_perf.reporters import aggregate, write_reports
from llm_perf.resilience import CircuitOpen, OperationFailed, execute, reset_resilience_state
from llm_perf.thresholds import evaluate


@pytest.fixture(autouse=True)
def clean_resilience_state():
    """Keep breaker/fault/metric state isolated between tests."""
    reset_resilience_state()
    metrics._CUSTOM.clear()
    metrics.reset_availability_state()
    yield
    reset_resilience_state()
    metrics._CUSTOM.clear()
    metrics.reset_availability_state()


def _policy(**kwargs) -> ResilienceConfig:
    """Build a resilience policy from plain YAML-shaped data."""
    return ResilienceConfig.model_validate(kwargs)


def test_resilience_config_parses_target_and_stage_overrides():
    """Targets can define resilience and pipeline stages can override parts."""
    cfg = PerfConfig.model_validate(
        {
            "targets": [
                {
                    "name": "rag",
                    "layer": "pipeline",
                    "dataset": "qa",
                    "resilience": {
                        "retry": {"attempts": 2, "backoff_s": 0.01},
                        "circuit_breaker": {"enabled": True, "failure_threshold": 3},
                    },
                    "pipeline": [
                        {
                            "stage": "retrieve",
                            "vector_store": {"type": "json", "path": "docs.jsonl"},
                            "resilience": {"fallback": {"type": "empty_context"}},
                        },
                        {"stage": "generate", "llm": {"provider": "openai"}},
                    ],
                }
            ],
            "scenarios": [{"name": "s", "target": "rag"}],
            "datasets": {"qa": {"source": "data.jsonl", "format": "jsonl"}},
        }
    )
    target = cfg.targets[0]
    assert target.resilience.retry.attempts == 2
    assert target.resilience.circuit_breaker.enabled is True
    assert target.pipeline[0].resilience.fallback.type == "empty_context"
    assert target.pipeline[1].resilience is None


def test_execute_retries_then_succeeds():
    """A configured retryable exception is retried before succeeding."""
    cfg = _policy(
        retry={
            "attempts": 2,
            "backoff_s": 0,
            "jitter_s": 0,
            "retry_on": ["exception"],
        }
    )
    calls = {"count": 0}

    def operation():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    out = execute(
        scenario="s",
        target="t",
        site="dep",
        resilience=cfg,
        operation=operation,
    )
    assert out.value == "ok"
    assert out.attempts == 2
    snap = metrics.resilience_snapshot()["s/t/dep"]
    assert snap["retry_attempts"] == 1
    assert snap["retry_exhausted"] == 0


def test_execute_exhausts_retries_then_static_fallback():
    """Retry exhaustion records metrics and invokes the configured fallback."""
    cfg = _policy(
        retry={
            "attempts": 1,
            "backoff_s": 0,
            "jitter_s": 0,
            "retry_on": ["exception"],
        },
        fallback={"type": "static_response", "response": "degraded"},
    )

    out = execute(
        scenario="s",
        target="t",
        site="dep",
        resilience=cfg,
        operation=lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )
    assert out.value == "degraded"
    assert out.degraded is True
    assert out.fallback_used is True
    snap = metrics.resilience_snapshot()["s/t/dep"]
    assert snap["retry_attempts"] == 1
    assert snap["retry_exhausted"] == 1
    assert snap["fallback_invocations"] == 1
    assert snap["fallback_successes"] == 1
    assert snap["degraded"] == 1


def test_provider_retry_exhaustion_is_recorded_when_fallback_succeeds():
    """Provider-adapter retries are preserved even though execute() did not retry."""
    cfg = _policy(
        fallback={"type": "static_response", "response": "degraded"},
    )
    failed = NormalizedResponse(text="", raw={"error": "down"}, retries=2, timed_out=True)

    out = execute(
        scenario="s",
        target="t",
        site="llm",
        resilience=cfg,
        operation=lambda: failed,
        result_failure=lambda resp: ("llm_error", "down"),
    )
    assert out.value == "degraded"
    snap = metrics.resilience_snapshot()["s/t/llm"]
    assert snap["retry_attempts"] == 2
    assert snap["retry_exhausted"] == 1
    assert snap["timeouts"] == 1


def test_circuit_breaker_opens_rejects_half_opens_and_closes():
    """Breaker state transitions are measured per call site."""
    cfg = _policy(
        circuit_breaker={
            "enabled": True,
            "failure_threshold": 1,
            "window_s": 30,
            "open_duration_s": 0.01,
            "half_open_calls": 1,
            "successes_to_close": 1,
        }
    )

    with pytest.raises(OperationFailed):
        execute(
            scenario="s",
            target="t",
            site="dep",
            resilience=cfg,
            operation=lambda: (_ for _ in ()).throw(RuntimeError("down")),
        )
    with pytest.raises(CircuitOpen):
        execute(
            scenario="s",
            target="t",
            site="dep",
            resilience=cfg,
            operation=lambda: "should-not-run",
        )

    time.sleep(0.02)
    out = execute(
        scenario="s",
        target="t",
        site="dep",
        resilience=cfg,
        operation=lambda: "recovered",
    )
    assert out.value == "recovered"
    snap = metrics.resilience_snapshot()["s/t/dep"]
    assert snap["circuit_state"] == "closed"
    assert snap["circuit_opens"] == 1
    assert snap["circuit_rejections"] == 1
    assert snap["circuit_half_opens"] == 1
    assert snap["circuit_closes"] == 1


def test_fault_injection_uses_fallback_and_is_measured():
    """Fault injection creates a deterministic failure without calling the op."""
    cfg = _policy(
        fault_injection={
            "enabled": True,
            "fault": "exception",
            "probability": 1.0,
            "after_s": 0,
            "duration_s": 30,
        },
        fallback={"type": "static_response", "response": "fallback"},
    )
    called = {"count": 0}

    def operation():
        called["count"] += 1
        return "primary"

    out = execute(
        scenario="s",
        target="t",
        site="dep",
        resilience=cfg,
        operation=operation,
    )
    assert called["count"] == 0
    assert out.value == "fallback"
    snap = metrics.resilience_snapshot()["s/t/dep"]
    assert snap["fault_injections"] == 1
    assert snap["fallback_successes"] == 1


def test_generate_stage_static_fallback_marks_degraded(monkeypatch):
    """A failed generate LLM result can degrade to a static response."""
    from llm_perf.pipelines import engine

    stage = PipelineStage.model_validate(
        {
            "stage": "generate",
            "llm": {"provider": "openai", "model": "primary"},
            "resilience": {"fallback": {"type": "static_response", "response": "backup answer"}},
        }
    )
    ctx = StageContext("q", {}, "s", {"_target_name": "rag", "_target_resilience": ResilienceConfig()})
    monkeypatch.setattr(
        engine,
        "_call_generate_llm",
        lambda *args, **kwargs: NormalizedResponse(text="", raw={"error": "down"}),
    )
    run_stage_generate(stage, ctx)
    assert ctx.output_text == "backup answer"
    assert ctx.degraded is True
    snap = metrics.resilience_snapshot()["s/rag/generate"]
    assert snap["fallback_successes"] == 1
    assert snap["degraded"] == 1


def test_llm_api_static_fallback(monkeypatch):
    """The direct LLM layer returns degraded output instead of an error."""
    target = LlmApiTarget.model_validate(
        {
            "name": "llm",
            "provider": "openai",
            "params": {"model": "primary"},
            "resilience": {"fallback": {"type": "static_response", "response": "fallback text"}},
        }
    )
    import llm_perf.layers.llm_api as layer_mod

    monkeypatch.setattr(
        layer_mod,
        "call_llm",
        lambda spec: NormalizedResponse(text="", raw={"error": "provider down"}),
    )
    result = LlmApiLayer(target, picker=None).run("prompt", {}, {}, "s")
    assert result.error is None
    assert result.output_text == "fallback text"
    assert result.degraded is True
    assert result.duration_s >= 0
    snap = metrics.resilience_snapshot()["s/llm/llm_api"]
    assert snap["fallback_successes"] == 1


def test_agent_alternate_llm_uses_fallback_model_and_params(monkeypatch):
    """Agent fallbacks get their own provider/model/params, not the primary spec."""
    from llm_perf.pipelines import engine

    res = _policy(
        fallback={
            "type": "alternate_llm",
            "llm": {
                "provider": "openai",
                "model": "backup-model",
                "params": {"temperature": 0.1},
            },
        }
    )
    ctx = StageContext("q", {}, "s", {"_target_name": "agent"})
    calls = []

    def fake_call(spec):
        calls.append(spec)
        if len(calls) == 1:
            return NormalizedResponse(text="", raw={"error": "primary down"})
        return NormalizedResponse(text="backup answer", raw={})

    monkeypatch.setattr(engine, "call_llm", fake_call)
    resp = engine._agent_llm_call(
        ctx,
        {"provider": "openai", "api_key_env": "PRIMARY_KEY"},
        "openai",
        {"model": "primary-model", "temperature": 0.7},
        [{"role": "user", "content": "q"}],
        0,
        res,
    )
    assert resp.text == "backup answer"
    assert calls[0].params["model"] == "primary-model"
    assert calls[1].params["model"] == "backup-model"
    assert calls[1].params["temperature"] == 0.1


def test_app_endpoint_connect_failure_uses_static_fallback():
    """Black-box endpoint failures can complete through a static fallback."""
    target = AppEndpointTarget.model_validate(
        {
            "name": "app",
            "base_url": "http://example.test",
            "endpoint": "/chat",
            "resilience": {"fallback": {"type": "static_response", "response": "endpoint fallback"}},
        }
    )

    class FailingHttp:
        def request(self, *args, **kwargs):
            raise requests.ConnectionError("connection refused")

    layer = AppEndpointLayer(target, picker=None)
    layer.http = FailingHttp()
    result = layer.run("prompt", {}, {}, "s")
    assert result.error is None
    assert result.output_text == "endpoint fallback"
    assert result.degraded is True
    snap = metrics.resilience_snapshot()["s/app/app_endpoint"]
    assert snap["fallback_successes"] == 1
    assert snap["degraded"] == 1


class _Entry:
    """Minimal Locust stats entry for report/threshold tests."""

    def __init__(self, name="s", method="", requests=10, failures=0):
        self.name = name
        self.method = method
        self.num_requests = requests
        self.num_failures = failures
        self.avg_response_time = 100.0
        self.min_response_time = 50.0
        self.max_response_time = 200.0

    def get_response_time_percentile(self, _pct):
        return 100.0


class _Stats:
    """Minimal Locust stats object."""

    def __init__(self):
        e = _Entry()
        self.entries = {"s:": e}
        self.total = SimpleNamespace(num_requests=10, num_failures=0)

    def get(self, name, method):
        return self.entries.get(f"{name}:{method}")


def test_resilience_metrics_are_reported_and_thresholded():
    """Scenario reports and SLO rules expose breaker/fallback/degraded metrics."""
    stats = _Stats()
    custom_stats = {
        ("s", "retry_exhausted"): {"sum": 2.0, "avg": 1.0},
        ("s", "circuit_open"): {"sum": 1.0, "avg": 1.0},
        ("s", "circuit_rejected"): {"sum": 3.0, "avg": 1.0},
        ("s", "fallback_invocations"): {"sum": 3.0, "avg": 1.0},
        ("s", "fallback_success"): {"sum": 2.0, "avg": 2 / 3},
        ("s", "degraded"): {"sum": 3.0, "avg": 0.3},
        ("s", "fault_injections"): {"sum": 1.0, "avg": 1.0},
    }
    cfg = PerfConfig.model_validate(
        {
            "targets": [{"name": "t", "provider": "openai", "prompts": ["hi"]}],
            "scenarios": [{"name": "s", "target": "t"}],
        }
    )
    report = aggregate(stats, cfg, 10.0, custom_stats=custom_stats)
    sc = report["scenarios"][0]
    assert sc["retry_exhausted_total"] == 2
    assert sc["circuit_open_total"] == 1
    assert sc["circuit_rejections_total"] == 3
    assert sc["fallback_invocations_total"] == 3
    assert sc["fallback_success_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert sc["degraded_rate"] == 0.3
    assert sc["fault_injections_total"] == 1

    rules = ThresholdsConfig.model_validate(
        {
            "scenarios": {
                "s": {
                    "max_retry_exhausted": 2,
                    "max_circuit_open_events": 1,
                    "max_circuit_rejections": 3,
                    "min_fallback_success_rate": 0.6,
                    "max_degraded_rate": 0.5,
                }
            }
        }
    )
    rep = evaluate(
        stats,
        rules,
        ["s"],
        duration_s=10,
        total_requests=10,
        total_failures=0,
        custom_stats=custom_stats,
    )
    assert rep.all_passed


def test_resilience_fields_render_in_html_report(tmp_path):
    """HTML reports expose both scenario rollups and call-site details."""
    stats = _Stats()
    cfg = PerfConfig.model_validate(
        {
            "targets": [{"name": "t", "provider": "openai", "prompts": ["hi"]}],
            "scenarios": [{"name": "s", "target": "t"}],
            "report": {"formats": ["html"], "output_dir": str(tmp_path)},
        }
    )
    report = aggregate(
        stats,
        cfg,
        10.0,
        custom_stats={
            ("s", "circuit_open"): {"sum": 1.0, "avg": 1.0},
            ("s", "fallback_success"): {"sum": 1.0, "avg": 1.0},
            ("s", "degraded"): {"sum": 1.0, "avg": 0.1},
        },
    )
    report["resilience"] = {
        "s/t/generate": {
            "circuit_state": "closed",
            "circuit_opens": 1.0,
            "circuit_half_opens": 1.0,
            "circuit_closes": 1.0,
            "circuit_rejections": 2.0,
            "retry_attempts": 2.0,
            "fallback_invocations": 1.0,
            "degraded": 1.0,
            "fault_injections": 1.0,
        }
    }
    [path] = write_reports(report, cfg)
    html = path.read_text()
    assert "Resilience" in html
    assert "s/t/generate" in html
    assert "closed" in html


def test_availability_samples_are_aggregated_by_run_and_phase():
    """Each request contributes hard, functional, and non-degraded outcomes."""
    now = time.time()
    metrics.record_availability(
        "s",
        available=True,
        functional_available=True,
        non_degraded_available=True,
        duration_s=0.100,
        timestamp=now - 5,
    )
    metrics.record_availability(
        "s",
        available=True,
        functional_available=True,
        non_degraded_available=False,
        duration_s=0.200,
        timestamp=now - 1,
    )
    metrics.record_availability(
        "s",
        available=False,
        functional_available=False,
        non_degraded_available=False,
        duration_s=0.300,
        timestamp=now,
    )

    whole_run = metrics.availability_summary()["s"]
    assert whole_run["requests"] == 3
    assert whole_run["availability"] == pytest.approx(2 / 3)
    assert whole_run["functional_availability"] == pytest.approx(2 / 3)
    assert whole_run["non_degraded_availability"] == pytest.approx(1 / 3)

    window = metrics.availability_window_summary(now - 6, now + 1)["s"]
    assert window["requests"] == 3
    assert window["availability"] == pytest.approx(2 / 3)
    assert window["functional_availability"] == pytest.approx(2 / 3)
    assert window["non_degraded_availability"] == pytest.approx(1 / 3)
    assert window["p50_ms"] == 200.0

    earlier = metrics.availability_window_summary(now - 10, now - 2)["s"]
    assert earlier["requests"] == 1
    assert earlier["non_degraded_availability"] == 1.0


def test_availability_metrics_are_reported_and_thresholded():
    """SLO fields are available in report aggregation and threshold evaluation."""
    stats = _Stats()
    custom_stats = {
        ("s", "availability"): {"count": 10, "sum": 9.0, "avg": 0.9},
        ("s", "functional_availability"): {"count": 10, "sum": 8.0, "avg": 0.8},
        ("s", "non_degraded_availability"): {"count": 10, "sum": 7.0, "avg": 0.7},
    }
    cfg = PerfConfig.model_validate(
        {
            "targets": [{"name": "t", "provider": "openai", "prompts": ["hi"]}],
            "scenarios": [{"name": "s", "target": "t"}],
        }
    )
    report = aggregate(stats, cfg, 10.0, custom_stats=custom_stats)
    sc = report["scenarios"][0]
    assert sc["availability"] == 0.9
    assert sc["functional_availability"] == 0.8
    assert sc["non_degraded_availability"] == 0.7
    assert report["totals"]["availability"] == 0.9

    rules = ThresholdsConfig.model_validate(
        {
            "scenarios": {
                "s": {
                    "min_availability": 0.89,
                    "min_functional_availability": 0.79,
                    "min_non_degraded_availability": 0.69,
                }
            }
        }
    )
    rep = evaluate(
        stats,
        rules,
        ["s"],
        duration_s=10,
        total_requests=10,
        total_failures=1,
        custom_stats=custom_stats,
    )
    assert rep.all_passed

    failing = ThresholdsConfig.model_validate({"scenarios": {"s": {"min_availability": 0.95}}})
    rep = evaluate(
        stats,
        failing,
        ["s"],
        duration_s=10,
        total_requests=10,
        total_failures=1,
        custom_stats=custom_stats,
    )
    assert not rep.all_passed
    assert rep.results[0].actual == 0.9


def test_availability_fields_render_in_html_report(tmp_path):
    """HTML exposes whole-run SLO metrics and per-phase windows."""
    stats = _Stats()
    cfg = PerfConfig.model_validate(
        {
            "targets": [{"name": "t", "provider": "openai", "prompts": ["hi"]}],
            "scenarios": [{"name": "s", "target": "t"}],
            "report": {"formats": ["html"], "output_dir": str(tmp_path)},
        }
    )
    report = aggregate(
        stats,
        cfg,
        10.0,
        custom_stats={
            ("s", "availability"): {"sum": 9.0, "avg": 0.9},
            ("s", "functional_availability"): {"sum": 8.0, "avg": 0.8},
            ("s", "non_degraded_availability"): {"sum": 7.0, "avg": 0.7},
        },
    )
    report["slo_windows"] = [
        {
            "label": "load",
            "scenarios": {
                "s": {
                    "requests": 10,
                    "availability": 0.9,
                    "functional_availability": 0.8,
                    "non_degraded_availability": 0.7,
                    "p50_ms": 100.0,
                    "p95_ms": 200.0,
                    "p99_ms": 300.0,
                }
            },
        }
    ]
    [path] = write_reports(report, cfg)
    html = path.read_text()
    assert "SLO windows" in html
    assert "Functional" in html
    assert "Non-degraded" in html
    assert "90.00%" in html
