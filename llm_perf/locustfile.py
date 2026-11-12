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


# Load-shape for profile/custom-stages runs. Only defined when configured so
# flat runs keep using plain -u/-r/-t locust args.
if _config.load.uses_shape:
    from locust import LoadTestShape

    from .workload import build_phases

    class LLMPerfShape(LoadTestShape):
        """Drives ramp-up / steady-state / ramp-down (and step profiles)."""

        def __init__(self):
            super().__init__()
            self._phases = build_phases(_config.load)
            self._idx = 0
            self._phase_start = 0.0
            self._last_label = None
            from .workload import _seconds

            self._run_time_cap = _seconds(_config.load.run_time, 0.0) or None

        def _current_phase(self):
            now = self.get_run_time()
            while self._idx < len(self._phases):
                p = self._phases[self._idx]
                if now < self._phase_start + p.duration_s:
                    if p.label != self._last_label:
                        self._last_label = p.label
                        print(
                            f"[llm-perf] phase {p.label}: users={p.target_users} "
                            f"spawn={p.spawn_rate}/s dur={p.duration_s:.0f}s"
                        )
                    return p
                self._phase_start += p.duration_s
                self._idx += 1
            return None

        def _capacity_breached(self) -> bool:
            if _config.load.profile != "capacity" or self.runner is None:
                return False
            try:
                t = self.runner.stats.total
                if t.num_requests < 20:
                    return False
                err = t.num_failures / max(t.num_requests, 1)
                return err > _config.load.capacity_max_error_rate
            except Exception:
                return False

        def tick(self):
            if self._capacity_breached():
                print("[llm-perf] capacity profile: error-rate breach, stopping early")
                return None
            if self._run_time_cap and self.get_run_time() >= self._run_time_cap:
                return None
            p = self._current_phase()
            if p is None:
                return None
            return p.target_users, p.spawn_rate


_test_start_ts = None
_resource_sampler = None
_lag_probe = None


@events.test_start.add_listener
def _on_test_start(environment, **kw):
    global _test_start_ts, _resource_sampler, _lag_probe
    import time

    _test_start_ts = time.time()
    oc = _config.observability.otel
    if oc.enabled:
        setup_otel(service_name=oc.service_name, endpoint=oc.endpoint, sample_ratio=oc.sample_ratio)
    # record_users is driven by User.on_start/on_stop
    if _config.report.collect_resources:
        from .resources import EventLoopLagProbe, HostSampler

        _resource_sampler = HostSampler(interval_s=5.0)
        _resource_sampler.start()
        _lag_probe = EventLoopLagProbe()
        _lag_probe.start()


@events.test_stop.add_listener
def _on_test_stop(environment, **kw):
    if _resource_sampler is not None:
        _resource_sampler.stop()
    if _lag_probe is not None:
        _lag_probe.stop()
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

    # stage stats + cost + exact custom samples from the request context
    from .metrics import custom_stats_snapshot, stage_stats_snapshot
    from .scenarios import get_context

    ctx = get_context()
    stage_stats = stage_stats_snapshot()
    custom_stats = custom_stats_snapshot()
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
            custom_stats=custom_stats,
        )
        # publish threshold gauges for live alerting dashboards
        from .metrics import record_threshold

        for r in trep.results:
            if r.passed is not None:
                record_threshold(r.name, r.passed)

    report = aggregate(stats, _config, dur or 0.0, trep, cost_by_scenario, custom_stats)
    _enrich_report(report, stage_stats)
    paths = write_reports(report, _config, trep)
    from . import history

    history.record_run(report, _config)
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

    # regression gate
    reg = report.get("regression") or {}
    if reg.get("regressions") and _config.report.regression.on_fail == "fail":
        print(f"[llm-perf] {len(reg['regressions'])} REGRESSION(S) — exiting nonzero")
        import os

        os.environ["LLM_PERF_THRESHOLDS_FAILED"] = "1"


def _enrich_report(report: dict, stage_stats: dict) -> None:
    """Attach load profile, stages, resources, analysis, comparison to report."""
    from . import analysis, history
    from .metrics import peak_users
    from .workload import schedule_summary

    report["load"] = {
        "profile": _config.load.profile,
        "users": _config.load.users,
        "spawn_rate": _config.load.spawn_rate,
        "duration": _config.load.duration,
        "run_time": _config.load.run_time,
    }
    report["phases"] = schedule_summary(_config.load)
    report["totals"]["peak_users"] = peak_users()
    report["stages"] = [{"scenario": sc, "stage": st, **vals} for (sc, st), vals in stage_stats.items()]

    resources = {}
    if _config.report.collect_resources:
        loadgen = _resource_sampler.summary() if _resource_sampler else {}
        if _lag_probe is not None:
            lag = _lag_probe.summary()
            if lag:
                loadgen["scheduler_lag"] = lag
        resources["loadgen"] = loadgen
    if _config.report.scrape:
        from .resources import collect_target_metrics

        resources["targets"] = collect_target_metrics(_config.report.scrape)
    if resources:
        report["resources"] = resources

    # generator-health warnings (CPU saturation / event-loop lag)
    from .resources import evaluate_resource_warnings

    warnings = evaluate_resource_warnings((resources.get("loadgen") or {}), _config.report.resource_warn)
    if warnings:
        report["warnings"] = warnings
        for w in warnings:
            print(
                f"[llm-perf] WARNING [{w['metric']}] {w['message']} "
                f"(value={w['value']}, threshold={w['threshold']})"
            )

    report["analysis"] = {
        "executive_summary": analysis.executive_summary(report),
        "slo": analysis.slo_rollup(report),
        "bottlenecks": analysis.identify_bottlenecks(report),
    }

    baseline = history.resolve_baseline(_config, report)
    if baseline:
        report["comparison"] = analysis.compare_runs(report, baseline)
        if _config.report.regression.enabled:
            report["regression"] = analysis.detect_regressions(
                report, baseline, _config.report.regression.thresholds_pct
            )
    else:
        report["comparison"] = None
        report["regression"] = {"checked": False, "regressions": []}
