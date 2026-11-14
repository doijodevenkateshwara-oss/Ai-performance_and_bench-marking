"""Generated locustfile entry point.

The CLI writes/uses this module so `locust -f llm_perf.locustfile` works with
the standard Locust CLI (web UI, --headless, --master/--worker).


Author: Venkateshwara Doijode
"""

from __future__ import annotations

import logging
import os

from locust import events

from .config import PerfConfig
from .otel.setup import setup_otel
from .otel.setup import shutdown as otel_shutdown
from .scenarios import build_users
from .security import set_policy, warn_untrusted

logger = logging.getLogger("llm_perf")

_config_path = os.environ.get("LLM_PERF_CONFIG")
if not _config_path:
    raise RuntimeError("LLM_PERF_CONFIG env var must point to a config YAML")

_config = PerfConfig.from_yaml(_config_path)
set_policy(_config.security)
warn_untrusted(_config)
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
            """LLMPerfShape(): precomputes the phase schedule for this run."""
            super().__init__()
            self._phases = build_phases(_config.load)
            self._idx = 0
            self._phase_start = 0.0
            self._last_label = None
            from .workload import _seconds

            self._run_time_cap = _seconds(_config.load.run_time, 0.0) or None

        def _current_phase(self):
            """Return the active phase for the current run time."""
            now = self.get_run_time()
            while self._idx < len(self._phases):
                p = self._phases[self._idx]
                if now < self._phase_start + p.duration_s:
                    if p.label != self._last_label:
                        self._last_label = p.label
                        logger.info(
                            "[llm-perf] phase %s: users=%s spawn=%s/s dur=%.0fs",
                            p.label,
                            p.target_users,
                            p.spawn_rate,
                            p.duration_s,
                        )
                    return p
                self._phase_start += p.duration_s
                self._idx += 1
            return None

        def _capacity_breached(self) -> bool:
            """True when the capacity profile's error ceiling is exceeded."""
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
            """LoadTestShape.tick: current (users, spawn_rate) or None to stop."""
            if self._capacity_breached():
                logger.warning("[llm-perf] capacity profile: error-rate breach, stopping early")
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
    """Locust test_start hook: OTel setup + resource samplers."""
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


def _is_worker(environment) -> bool:
    """True on distributed workers — they must not write reports/alerts."""
    try:
        from locust.runners import WorkerRunner

        return isinstance(environment.runner, WorkerRunner)
    except Exception:
        return False


@events.test_stop.add_listener
def _on_test_stop(environment, **kw):
    """Locust test_stop hook: stop samplers, write reports, gate exit code."""
    if _resource_sampler is not None:
        _resource_sampler.stop()
    if _lag_probe is not None:
        _lag_probe.stop()
    if _is_worker(environment):
        return  # only the master/local runner aggregates reports + alerts
    _run_reports_and_thresholds(environment)


@events.quitting.add_listener
def _on_quitting(environment, **kw):
    """Locust quitting hook: flush/shutdown OTel exporters."""
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
    _enrich_report(report, stage_stats, cost_by_scenario)
    paths = write_reports(report, _config, trep)
    from . import history

    history.record_run(report, _config)
    for p in paths:
        logger.info("[llm-perf] report written: %s", p)

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
            logger.info("[llm-perf] alerts sent via: %s", ", ".join(set(notified)))

    # exit code for CI — three-state rules: FAIL breaches + NOT_EVALUATED policy
    if trep:
        if trep.should_fail_ci:
            reasons = []
            if trep.failed and _config.thresholds.on_fail == "fail":
                reasons.append(f"{len(trep.failed)} threshold failure(s)")
            if trep.not_evaluated and _config.thresholds.on_not_evaluated == "fail":
                reasons.append(
                    f"{len(trep.not_evaluated)} rule(s) not evaluated " "(thresholds.on_not_evaluated=fail)"
                )
            logger.error("[llm-perf] THRESHOLD FAILURES (%s) — exiting nonzero", "; ".join(reasons))
            os.environ["LLM_PERF_THRESHOLDS_FAILED"] = "1"
        if trep.should_warn:
            names = ", ".join(r.name for r in trep.not_evaluated[:10])
            logger.warning(
                "[llm-perf] WARNING: %s threshold rule(s) not evaluated (no data): %s",
                len(trep.not_evaluated),
                names,
            )

    # regression gate
    reg = report.get("regression") or {}
    if reg.get("regressions") and _config.report.regression.on_fail == "fail":
        logger.error("[llm-perf] %s REGRESSION(S) — exiting nonzero", len(reg["regressions"]))
        os.environ["LLM_PERF_THRESHOLDS_FAILED"] = "1"


def _enrich_report(report: dict, stage_stats: dict, cost_by_scenario: dict | None = None) -> None:
    """Attach load profile, stages, resources, analysis, comparison to report."""
    from . import analysis, history, manifest
    from .metrics import peak_users
    from .workload import schedule_summary

    # reproducibility manifest — provenance for cross-run comparisons
    report["manifest"] = manifest.build(_config, _config_path)
    report["run_id"] = report["manifest"]["run_id"]

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

    # per-request latency composition (percentile-safe: shares computed per
    # request, then aggregated) + cost broken down by call-site stage
    from .metrics import breakdown_summary

    comp = breakdown_summary()
    if comp:
        report["composition"] = comp
    _attach_cost_breakdown(report, cost_by_scenario)

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
            logger.warning(
                "[llm-perf] WARNING [%s] %s (value=%s, threshold=%s)",
                w["metric"],
                w["message"],
                w["value"],
                w["threshold"],
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


def _attach_cost_breakdown(report: dict, cost_by_scenario: dict | None) -> None:
    """Attach report['cost_breakdown']: {scenario: {category: usd}}.

    Canonical categories come from record_cost stage tags; "unattributed" is
    the scenario total minus categorized spend (catches embedding spend or
    any call site not yet tagged). Never exceeds the recorded total.
    """
    from .metrics import cost_stage_snapshot

    cost_stages = cost_stage_snapshot()
    if not cost_stages and not cost_by_scenario:
        return
    cats = ("generation", "reranking", "embedding", "agent", "judge", "retry")
    breakdown: dict[str, dict] = {}
    for sc, stages in cost_stages.items():
        entry = {c: round(stages.get(c, 0.0), 6) for c in cats if stages.get(c)}
        if entry:
            breakdown[sc] = entry
    for sc, total in (cost_by_scenario or {}).items():
        if sc == "__total__" or total is None:
            continue
        entry = breakdown.setdefault(sc, {})
        attributed = sum(entry.values())
        remainder = round(max(total - attributed, 0.0), 6)
        if remainder > 1e-9:
            entry["unattributed"] = remainder
        elif not entry:
            breakdown.pop(sc, None)
    if breakdown:
        report["cost_breakdown"] = breakdown
