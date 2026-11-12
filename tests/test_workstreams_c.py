"""Tests for workload profiles, analysis, history, and resources.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json

import pytest

from llm_perf import analysis, history
from llm_perf.config import LoadConfig, PerfConfig, ReportConfig
from llm_perf.resources import HostSampler, scrape_prometheus
from llm_perf.workload import build_phases, schedule_summary


def _load(**kw) -> LoadConfig:
    """Load a YAML config string into PerfConfig."""
    return LoadConfig(users=kw.pop("users", 10), spawn_rate=kw.pop("spawn_rate", 2.0), **kw)


# ----------------------------- workload profiles -----------------------------


def test_smoke_profile_phases():
    """smoke profile yields ramp/hold/down phases."""
    phases = build_phases(_load(profile="smoke", duration="2m", ramp_up="30s", ramp_down="15s"))
    labels = [p.label for p in phases]
    assert labels == ["ramp_up", "steady", "ramp_down"]
    assert phases[0].target_users == 10
    assert phases[-1].target_users == 0


def test_load_profile_default_ramp():
    """load profile derives ramp from users/spawn_rate."""
    phases = build_phases(_load(profile="load", users=10, spawn_rate=2.0, duration="5m"))
    # default ramp_up = users/spawn_rate = 5s
    assert phases[0].label == "ramp_up"
    assert phases[0].duration_s == pytest.approx(5.0)
    assert phases[1].label == "steady"


def test_stress_profile_steps():
    """stress profile steps through increasing user levels."""
    phases = build_phases(_load(profile="stress", users=8, steps=4, step_duration="1m"))
    step_phases = [p for p in phases if p.label.startswith("step_")]
    assert len(step_phases) == 4
    assert [p.target_users for p in step_phases] == [2, 4, 6, 8]


def test_scalability_profile():
    """scalability holds each step plus a final steady phase."""
    phases = build_phases(
        _load(profile="scalability", users=10, step_users=5, step_duration="1m", duration="5m")
    )
    steps = [p for p in phases if p.label.startswith("step_")]
    assert [p.target_users for p in steps] == [5, 10]
    assert phases[-2].label == "steady"  # final hold


def test_capacity_profile_has_steps_and_down():
    """capacity profile steps then ramps down."""
    phases = build_phases(_load(profile="capacity", users=10, step_users=2, step_duration="30s"))
    assert any(p.target_users == 0 for p in phases)  # ramp down


def test_spike_profile():
    """spike profile spikes to users then returns to baseline."""
    phases = build_phases(_load(profile="spike", users=10, duration="1m", baseline_duration="30s"))
    labels = [p.label for p in phases]
    assert "spike_ramp" in labels and "spike_hold" in labels and "baseline_hold" in labels


def test_custom_stages():
    """Explicit stages become phases plus a ramp-down."""
    load = _load(
        stages=[
            {"users": 5, "spawn_rate": 1.0, "duration": "30s"},
            {"users": 10, "spawn_rate": 2.0, "duration": "1m"},
        ]
    )
    phases = build_phases(load)
    assert phases[0].target_users == 5
    assert phases[1].target_users == 10
    assert phases[-1].target_users == 0


def test_no_profile_no_shape():
    """No profile/stages -> empty phase list (flat locust args)."""
    assert build_phases(_load()) == []
    assert _load().uses_shape is False
    assert _load(profile="load").uses_shape is True


def test_schedule_summary_serializable():
    """schedule_summary emits JSON-safe phase dicts."""
    summary = schedule_summary(_load(profile="smoke", duration="1m"))
    json.dumps(summary)  # must not raise
    assert summary[0]["label"] == "ramp_up"


def test_duration_parsing():
    """_seconds parses s/m/h/d and compound durations."""
    phases = build_phases(_load(profile="smoke", duration="90s", ramp_up="1m", ramp_down="30s"))
    assert phases[1].duration_s == pytest.approx(90.0)
    assert phases[0].duration_s == pytest.approx(60.0)


# ----------------------------- analysis --------------------------------------


def _report(**over) -> dict:
    """Build a minimal aggregate report dict for analysis tests."""
    base = {
        "generated_at": "2024-01-01T00:00:00",
        "duration_s": 60.0,
        "load": {"profile": "load"},
        "totals": {"requests": 100, "failures": 2, "error_rate": 0.02, "peak_users": 10},
        "scenarios": [
            {
                "scenario": "chat",
                "requests": 100,
                "failures": 2,
                "error_rate": 0.02,
                "avg_ms": 500.0,
                "p50_ms": 400.0,
                "p90_ms": 800.0,
                "p95_ms": 1000.0,
                "p99_ms": 1500.0,
                "rps": 1.67,
                "ttft_p95_ms": 300.0,
                "tpot_p95_ms": 20.0,
                "itl_p95_ms": 25.0,
                "gen_duration_p95_ms": 700.0,
                "queue_wait_p95_ms": 50.0,
                "tokens_per_sec_avg": 40.0,
                "input_tokens_per_sec_avg": 200.0,
                "total_tokens": 5000,
                "quality_score_avg": 0.8,
                "context_recall_avg": 0.9,
                "retries_total": 3.0,
                "timeouts": 1,
                "cost_usd": 0.5,
                "cost_per_success_usd": 0.005,
            }
        ],
        "stages": [
            {
                "scenario": "chat",
                "stage": "retrieve",
                "count": 100,
                "avg_ms": 100,
                "p50_ms": 90,
                "p95_ms": 200,
                "max_ms": 400,
            },
            {
                "scenario": "chat",
                "stage": "generate",
                "count": 100,
                "avg_ms": 800,
                "p50_ms": 700,
                "p95_ms": 1500,
                "max_ms": 3000,
            },
        ],
        "thresholds": {
            "rules": [
                {
                    "rule": "r1",
                    "metric": "m",
                    "comparator": "<=",
                    "threshold": 1,
                    "actual": 0.5,
                    "passed": True,
                    "scope": "global",
                },
                {
                    "rule": "r2",
                    "metric": "m",
                    "comparator": "<=",
                    "threshold": 1,
                    "actual": 2.0,
                    "passed": False,
                    "scope": "scenario",
                },
            ]
        },
    }
    base.update(over)
    return base


def test_executive_summary():
    """executive_summary produces verdict/findings/slo rollup."""
    es = analysis.executive_summary(_report())
    assert es["verdict"] == "FAIL"  # one threshold rule failed
    assert es["headline_scenario"] == "chat"
    assert es["slo"]["fail"] == 1
    assert es["slo"]["pass"] == 1
    assert any("chat" in f for f in es["findings"])


def test_executive_summary_no_data():
    """Empty runs get a NO DATA verdict."""
    es = analysis.executive_summary({"totals": {}, "scenarios": [], "thresholds": {"rules": []}})
    assert es["verdict"] == "NO DATA"


def test_identify_bottlenecks_decode_bound():
    """Decode-dominated scenarios flag decode_bound."""
    b = analysis.identify_bottlenecks(_report())
    types = {x["type"] for x in b}
    assert "decode_bound" in types or "stage" in types


def test_identify_bottlenecks_reliability():
    """High retry/timeout ratios flag reliability."""
    rep = _report()
    rep["scenarios"][0].update({"retries_total": 10.0, "timeouts": 5, "requests": 100})
    b = analysis.identify_bottlenecks(rep)
    assert any(x["type"] == "reliability" for x in b)


def test_compare_runs():
    """compare_runs computes per-metric deltas vs baseline."""
    cur = _report()
    base = _report()
    base["scenarios"][0]["p95_ms"] = 2000.0  # baseline slower -> current better
    comp = analysis.compare_runs(cur, base)
    assert comp is not None
    p95 = next(m for m in comp["metrics"] if m["metric"] == "p95_ms")
    assert p95["delta"]["direction"] == "better"


def test_compare_runs_missing_scenario():
    """Scenarios missing on one side are noted."""
    cur = _report()
    base = _report()
    base["scenarios"][0]["scenario"] = "other"
    comp = analysis.compare_runs(cur, base)
    assert comp["per_scenario"]


def test_detect_regressions_flags_worse():
    """Worse-than-allowed metrics are flagged."""
    cur = _report()
    base = _report()
    cur["scenarios"][0]["p95_ms"] = 2000.0  # +100% vs baseline 1000
    reg = analysis.detect_regressions(cur, base)
    assert reg["checked"] is True
    assert any(r["metric"] == "p95_ms" for r in reg["regressions"])


def test_detect_regressions_clean():
    """Within-tolerance changes produce no regressions."""
    reg = analysis.detect_regressions(_report(), _report())
    assert reg["checked"] is True
    assert reg["regressions"] == []


def test_detect_regressions_no_baseline():
    """No baseline -> checked=False."""
    reg = analysis.detect_regressions(_report(), None)
    assert reg["checked"] is False


# ----------------------------- history ---------------------------------------


def test_history_record_and_previous(tmp_path):
    """record_run persists reports; previous_run finds them."""
    cfg = PerfConfig(report=ReportConfig(output_dir=str(tmp_path), history=True))
    rep = _report()
    rep["generated_at"] = "2024-01-01T00:00:00"
    path = history.record_run(rep, cfg)
    assert path is not None and path.exists()

    rep2 = _report()
    rep2["generated_at"] = "2024-01-02T00:00:00"
    rep2["totals"] = {"requests": 50, "failures": 0, "error_rate": 0.0}
    history.record_run(rep2, cfg)

    prev = history.previous_run(cfg, before_ts="2024-01-03T00:00:00")
    assert prev["generated_at"] == "2024-01-02T00:00:00"
    # strictly before current
    prev2 = history.previous_run(cfg, before_ts="2024-01-02T00:00:00")
    assert prev2["generated_at"] == "2024-01-01T00:00:00"


def test_history_disabled(tmp_path):
    """history=false skips persistence."""
    cfg = PerfConfig(report=ReportConfig(output_dir=str(tmp_path), history=False))
    assert history.record_run(_report(), cfg) is None


def test_mark_and_resolve_baseline(tmp_path):
    """mark_baseline + resolve_baseline(designated) round-trip."""
    cfg = PerfConfig(report=ReportConfig(output_dir=str(tmp_path), history=True, baseline="designated"))
    rep = _report()
    rep["generated_at"] = "2024-01-01T00:00:00"
    history.record_run(rep, cfg)
    assert history.mark_baseline(cfg) is not None

    current = _report()
    current["generated_at"] = "2024-01-02T00:00:00"
    resolved = history.resolve_baseline(cfg, current)
    assert resolved["generated_at"] == "2024-01-01T00:00:00"


def test_resolve_baseline_previous(tmp_path):
    """resolve_baseline(previous) picks the prior run."""
    cfg = PerfConfig(report=ReportConfig(output_dir=str(tmp_path), history=True, baseline="previous"))
    rep = _report()
    rep["generated_at"] = "2024-01-01T00:00:00"
    history.record_run(rep, cfg)
    current = _report()
    current["generated_at"] = "2024-01-02T00:00:00"
    resolved = history.resolve_baseline(cfg, current)
    assert resolved is not None


def test_resolve_baseline_none(tmp_path):
    """baseline=none resolves to None."""
    cfg = PerfConfig(report=ReportConfig(output_dir=str(tmp_path), baseline="none"))
    assert history.resolve_baseline(cfg, _report()) is None


# ----------------------------- resources -------------------------------------


def test_host_sampler_summary():
    """HostSampler aggregates CPU/mem samples (psutil-mocked or empty)."""
    s = HostSampler(interval_s=0.01)
    s._sample()
    s._sample()
    summary = s.summary()
    assert summary["samples"] == 2
    assert "cpu_percent_avg" in summary


def test_scrape_prometheus(tmp_path):
    """scrape_prometheus sums named metrics from exposition text."""
    body = """
# HELP http_requests_total Total requests
http_requests_total{code="200"} 100
http_requests_total{code="500"} 5
process_cpu_seconds_total 12.5
"""
    p = tmp_path / "metrics.txt"
    p.write_text(body)

    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        """Stub HTTP handler serving a Prometheus exposition."""

        def do_GET(self):
            """Serve canned metrics text."""
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            """Silence handler logging."""

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = scrape_prometheus(f"http://127.0.0.1:{srv.server_port}/metrics", ["http_requests_total"])
        assert out["http_requests_total"]["sum"] == pytest.approx(105.0)
        assert out["http_requests_total"]["samples"] == 2
        # unlisted metric excluded
        assert "process_cpu_seconds_total" not in out
    finally:
        srv.shutdown()


def test_scrape_prometheus_error():
    """Unreachable scrape URLs return an error dict."""
    out = scrape_prometheus("http://127.0.0.1:1/metrics", [])
    assert "error" in out
