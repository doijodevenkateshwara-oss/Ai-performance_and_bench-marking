"""Generated locustfile entry point.

The CLI writes/uses this module so `locust -f llm_perf.locustfile` works with
the standard Locust CLI (web UI, --headless, --master/--worker).
"""

from __future__ import annotations

import os

from locust import events

from .config import PerfConfig
from .otel.setup import setup_otel
from .otel.setup import shutdown as otel_shutdown
from .scenarios import build_users

_config_path = os.environ.get("LLM_PERF_CONFIG")
if not _config_path:
    raise RuntimeError("LLM_PERF_CONFIG env var must point to a config YAML")

_config = PerfConfig.from_yaml(_config_path)
_users = build_users(_config)

# expose generated user classes at module level for locust discovery
for _u in _users:
    globals()[_u.__name__] = _u


_test_start_ts = None


@events.test_start.add_listener
def _on_test_start(environment, **kw):
    global _test_start_ts
    import time

    _test_start_ts = time.time()
    oc = _config.observability.otel
    if oc.enabled:
        setup_otel(service_name=oc.service_name, endpoint=oc.endpoint, sample_ratio=oc.sample_ratio)
    # record_users is driven by User.on_start/on_stop


@events.test_stop.add_listener
def _on_test_stop(environment, **kw):
    _run_reports_and_thresholds(environment)


@events.quitting.add_listener
def _on_quitting(environment, **kw):
    otel_shutdown()


def _run_reports_and_thresholds(environment) -> None:
    """Aggregate stats, evaluate thresholds, write reports, send alerts."""
    from . import thresholds as th
    from .reporters import aggregate, write_reports

    stats = environment.stats
    import time as _time

    dur = (_time.time() - _test_start_ts) if _test_start_ts else 0.0

    total_req = stats.total.num_requests if hasattr(stats, "total") else 0
    total_fail = stats.total.num_failures if hasattr(stats, "total") else 0

    # stage stats + cost from the request context
    from .metrics import stage_stats_snapshot
    from .scenarios import get_context

    ctx = get_context()
    stage_stats = stage_stats_snapshot()
    cost_by_scenario = dict(getattr(ctx, "cost_by_scenario", {}) or {})
    if cost_by_scenario:
        cost_by_scenario["__total__"] = sum(cost_by_scenario.values())

    trep = None
    if _config.thresholds:
        trep = th.evaluate(
            stats,
            _config.thresholds,
            [s.name for s in _config.scenarios],
            duration_s=dur or 0.0,
            total_requests=total_req,
            total_failures=total_fail,
            cost_by_scenario=cost_by_scenario,
            stage_stats=stage_stats,
        )
        # publish threshold gauges for live alerting dashboards
        from .metrics import record_threshold

        for r in trep.results:
            if r.passed is not None:
                record_threshold(r.name, r.passed)

    report = aggregate(stats, _config, dur or 0.0, trep, cost_by_scenario)
    paths = write_reports(report, _config, trep)
    for p in paths:
        print(f"[llm-perf] report written: {p}")

    # alerts
    if _config.thresholds and _config.thresholds.alerts:
        summary = {
            "requests": total_req,
            "failures": total_fail,
            "duration_s": dur,
            "scenarios": len(_config.scenarios),
        }
        notified = th.send_alerts(
            _config.thresholds.alerts, trep or th.ThresholdReport(), "test_complete", summary
        )
        if notified:
            print(f"[llm-perf] alerts sent via: {', '.join(set(notified))}")

    # exit code for CI
    if trep and not trep.all_passed and _config.thresholds.on_fail == "fail":
        print("[llm-perf] THRESHOLD FAILURES — exiting nonzero")
        import os

        os.environ["LLM_PERF_THRESHOLDS_FAILED"] = "1"
