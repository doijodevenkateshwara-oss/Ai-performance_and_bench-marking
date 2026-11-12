"""SLO threshold evaluation, verdicts, and webhook alerts.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field

from .config import ThresholdRule, ThresholdsConfig


@dataclass
class RuleResult:
    """Outcome of one SLO rule evaluation."""

    name: str
    metric: str
    comparator: str  # <= or >=
    threshold: float
    actual: float | None
    passed: bool | None  # None = skipped (no data)
    scope: str = "scenario"  # global | scenario

    @property
    def status(self) -> str:
        """Three-state verdict: pass | fail | not_evaluated."""
        if self.passed is True:
            return "pass"
        if self.passed is False:
            return "fail"
        return "not_evaluated"


@dataclass
class ThresholdReport:
    """All rule results with aggregate pass/fail helpers."""

    results: list[RuleResult] = field(default_factory=list)
    on_fail: str = "fail"  # policy copied from ThresholdsConfig at evaluate() time
    on_not_evaluated: str = "pass"

    @property
    def all_passed(self) -> bool:
        """True when no rule explicitly failed (skips count as pass)."""
        return all(r.passed is not False for r in self.results)

    @property
    def failed(self) -> list[RuleResult]:
        """Rules that evaluated to FAIL."""
        return [r for r in self.results if r.passed is False]

    @property
    def not_evaluated(self) -> list[RuleResult]:
        """Rules with no data — neither pass nor fail."""
        return [r for r in self.results if r.passed is None]

    @property
    def should_fail_ci(self) -> bool:
        """True when CI should exit nonzero under the configured policies."""
        if self.failed and self.on_fail == "fail":
            return True
        return bool(self.not_evaluated) and self.on_not_evaluated == "fail"

    @property
    def should_warn(self) -> bool:
        """True when unevaluated rules should surface as a warning only."""
        return bool(self.not_evaluated) and self.on_not_evaluated == "warn"


# metric name -> (stats lookup kind)
_PCT_RULES = [("p50_ms", 50), ("p90_ms", 90), ("p95_ms", 95), ("p99_ms", 99)]
# custom metric -> (ThresholdRule field, exact-sample name, Locust stat method)
_CUSTOM_P95_RULES = [
    ("ttft_p95_ms", "ttft_ms", "TTFT"),
    ("tpot_p95_ms", "tpot_ms", "TPOT"),
    ("itl_p95_ms", "itl_p95_ms", "ITL_P95"),
    ("gen_p95_ms", "gen_duration_ms", "GEN_DURATION"),
    ("queue_p95_ms", "queue_wait_ms", "QUEUE_WAIT"),
]
_GAUGE_RULES = [
    ("max_error_rate", "error_rate", "<="),
    ("max_retries", "retries", "<="),
    ("max_timeouts", "timeouts", "<="),
    ("min_throughput_rps", "throughput_rps", ">="),
    ("min_quality_score", "quality_score", ">="),
    ("min_quality_pass_rate", "quality_pass_rate", ">="),
    ("min_context_recall", "context_recall", ">="),
    ("max_cost_usd", "cost_usd", "<="),
    ("max_cost_per_success_usd", "cost_per_success_usd", "<="),
    ("min_agent_success_rate", "agent_success_rate", ">="),
    ("max_agent_tool_error_rate", "agent_tool_error_rate", "<="),
    ("max_agent_loops_avg", "agent_loops_avg", "<="),
]


def _request_entries(stats, scenario: str | None):
    """Yield request-counting entries (excludes custom-stat entries like TTFT)."""
    try:
        for e in stats.entries.values():
            if e.method not in ("", "POST", "GET"):
                continue
            if scenario is None or e.name == scenario:
                yield e
    except Exception:
        return


def _entry_for(stats, scenario: str, method: str):
    """Fetch a (scenario, method) stats entry safely."""
    try:
        return stats.get(scenario, method)
    except Exception:
        return None


def evaluate(
    stats,
    thresholds: ThresholdsConfig | None,
    scenario_names: list[str],
    duration_s: float,
    total_requests: int,
    total_failures: int,
    cost_by_scenario: dict | None = None,
    stage_stats: dict | None = None,
    custom_stats: dict | None = None,
) -> ThresholdReport:
    """Evaluate all rules against final Locust stats + exact custom samples."""
    rep = ThresholdReport()
    if thresholds is None:
        return rep
    cost_by_scenario = cost_by_scenario or {}
    stage_stats = stage_stats or {}
    custom_stats = custom_stats or {}

    def check_rule(rule: ThresholdRule, scope: str, scenario: str | None):
        """Evaluate every set field of one ThresholdRule for a scope."""
        label = f"{scope}:{scenario}" if scenario else scope
        # latency percentiles — use the scenario's request entry (method "" aggregate)
        entry = _aggregate_entry(stats, scenario) if scenario else _global_entry(stats)
        for field_name, pct in _PCT_RULES:
            limit = getattr(rule, field_name)
            if limit is None:
                continue
            actual = _percentile(entry, pct) if entry else None
            rep.results.append(
                RuleResult(
                    name=f"{label}.{field_name}",
                    metric=f"p{int(pct)}_latency_ms",
                    comparator="<=",
                    threshold=limit,
                    actual=actual,
                    passed=(actual <= limit) if actual is not None else None,
                    scope=scope,
                )
            )
        # p95 rules on custom metrics (exact samples preferred, stats entry fallback)
        for field_name, metric, method in _CUSTOM_P95_RULES:
            limit = getattr(rule, field_name)
            if limit is None:
                continue
            se = custom_stats.get((scenario or "", metric)) if scenario else None
            e = _entry_for(stats, scenario or "", method) if scenario else None
            actual = se["p95"] if se else (_percentile(e, 95) if e else None)
            rep.results.append(
                RuleResult(
                    name=f"{label}.{field_name}",
                    metric=f"{method.lower()}_p95_ms",
                    comparator="<=",
                    threshold=limit,
                    actual=actual,
                    passed=(actual <= limit) if actual is not None else None,
                    scope=scope,
                )
            )
        # gauges
        for field_name, metric, comp in _GAUGE_RULES:
            limit = getattr(rule, field_name)
            if limit is None:
                continue
            actual = _gauge_value(
                metric,
                stats,
                scenario,
                duration_s,
                total_requests,
                total_failures,
                cost_by_scenario,
                stage_stats,
                custom_stats,
            )
            passed = None if actual is None else actual <= limit if comp == "<=" else actual >= limit
            rep.results.append(
                RuleResult(
                    name=f"{label}.{field_name}",
                    metric=metric,
                    comparator=comp,
                    threshold=limit,
                    actual=actual,
                    passed=passed,
                    scope=scope,
                )
            )
        # per-stage rules
        for stage_name, stage_rule in (rule.stages or {}).items():
            for m_name, limit in stage_rule.items():
                # stage_rule like {"p95_ms": 300}
                actual = _stage_percentile(stage_stats, scenario, stage_name, m_name)
                rep.results.append(
                    RuleResult(
                        name=f"{label}.stage.{stage_name}.{m_name}",
                        metric=f"stage.{stage_name}.{m_name}",
                        comparator="<=",
                        threshold=limit,
                        actual=actual,
                        passed=(actual <= limit) if actual is not None else None,
                        scope=scope,
                    )
                )

    if thresholds.global_:
        check_rule(thresholds.global_, "global", None)
    for sc_name, rule in (thresholds.scenarios or {}).items():
        if sc_name in scenario_names or not scenario_names:
            check_rule(rule, "scenario", sc_name)
    rep.on_fail = thresholds.on_fail
    rep.on_not_evaluated = thresholds.on_not_evaluated
    return rep


def _aggregate_entry(stats, scenario: str):
    # the request entry is named scenario+method; use aggregate across methods
    """Pick the dominant request entry for a scenario."""
    entries = []
    try:
        for e in stats.entries.values():
            if e.name == scenario:
                entries.append(e)
    except Exception:
        pass
    if not entries:
        return None
    # combine: return synthetic with weighted percentiles approximated by max entry
    return max(entries, key=lambda e: e.num_requests)


def _global_entry(stats):
    """Return stats.total as the global-scope entry."""
    try:
        return stats.total
    except Exception:
        return None


def _percentile(entry, pct: float) -> float | None:
    """Response-time percentile from a stats entry (API-tolerant)."""
    if entry is None:
        return None
    try:
        return entry.get_response_time_percentile(pct)
    except Exception:
        # older locust API
        try:
            return entry.percentile(pct)
        except Exception:
            return None


def _snap_avg(custom_stats, scenario, metric):
    """Average of exact samples for (scenario, metric), or None."""
    se = custom_stats.get((scenario or "", metric)) if custom_stats else None
    return se["avg"] if se else None


def _snap_sum(custom_stats, scenario, metric):
    """Sum of exact samples for (scenario, metric), or None."""
    se = custom_stats.get((scenario or "", metric)) if custom_stats else None
    return se["sum"] if se else None


def _gauge_value(
    metric,
    stats,
    scenario,
    duration_s,
    total_requests,
    total_failures,
    cost_by_scenario,
    stage_stats,
    custom_stats=None,
):
    """Compute the actual value for one gauge-style threshold metric."""
    try:
        if metric == "error_rate":
            # scope the denominator: scenario rules must use that scenario's
            # request count, not the global total
            fails = reqs = 0
            for e in _request_entries(stats, scenario):
                reqs += e.num_requests
                fails += e.num_failures
            if reqs <= 0:
                return 0.0
            return fails / reqs
        if metric == "throughput_rps":
            reqs = (
                total_requests
                if scenario is None
                else sum(e.num_requests for e in _request_entries(stats, scenario))
            )
            return reqs / max(duration_s, 1e-9)
        if metric == "quality_score":
            v = _snap_avg(custom_stats, scenario, "quality_score")
            if v is not None:
                return v
            e = _entry_for(stats, scenario or "", "QUALITY_SCORE") if scenario else None
            if e is None or e.num_requests == 0:
                return None
            return (e.avg_response_time or 0) / 1000.0
        if metric == "quality_pass_rate":
            v = _snap_avg(custom_stats, scenario, "quality_pass")
            if v is not None:
                return v
            p = _entry_for(stats, scenario or "", "QUALITY_SCORE") if scenario else None
            # fallback: pass rate approximated as avg>=0.5
            if p is None or p.num_requests == 0:
                return None
            return 1.0 if (p.avg_response_time or 0) / 1000.0 >= 0.5 else 0.0
        if metric == "context_recall":
            v = _snap_avg(custom_stats, scenario, "context_recall")
            if v is not None:
                return v
            e = _entry_for(stats, scenario or "", "CONTEXT_RECALL") if scenario else None
            if e is None or e.num_requests == 0:
                return None
            return (e.avg_response_time or 0) / 1000.0
        if metric == "cost_usd":
            return cost_by_scenario.get(scenario or "__total__")
        if metric == "cost_per_success_usd":
            return _cost_per_success(stats, scenario, cost_by_scenario)
        if metric == "agent_success_rate":
            return _snap_avg(custom_stats, scenario, "agent_success")
        if metric == "agent_tool_error_rate":
            tc = _snap_sum(custom_stats, scenario, "agent_tool_calls")
            te = _snap_sum(custom_stats, scenario, "agent_tool_errors")
            if not tc:
                return None
            return (te or 0.0) / tc
        if metric == "agent_loops_avg":
            return _snap_avg(custom_stats, scenario, "agent_loops")
        if metric == "retries":
            e = _entry_for(stats, scenario or "", "RETRIES") if scenario else None
            if e is None or e.num_requests == 0:
                return 0.0
            # logged value = retries per response; total = avg * count
            return (e.avg_response_time or 0.0) * e.num_requests
        if metric == "timeouts":
            e = _entry_for(stats, scenario or "", "TIMEOUTS") if scenario else None
            return float(e.num_requests) if e is not None else 0.0
    except Exception:
        return None
    return None


def _cost_per_success(stats, scenario, cost_by_scenario):
    """Total scenario cost / successful requests."""
    cost = cost_by_scenario.get(scenario or "__total__")
    if not cost:
        return None
    reqs = fails = 0
    try:
        for e in stats.entries.values():
            if (scenario is None or e.name == scenario) and e.method in ("", "POST", "GET"):
                reqs += e.num_requests
                fails += e.num_failures
    except Exception:
        return None
    successes = max(reqs - fails, 1)
    return cost / successes


def _stage_percentile(stage_stats, scenario, stage_name, metric_name):
    """Percentile of stage-duration samples for (scenario, stage)."""
    key = (scenario, stage_name)
    d = stage_stats.get(key) or stage_stats.get((None, stage_name))
    if not d:
        return None
    if metric_name == "p95_ms":
        return d.get("p95_ms")
    if metric_name == "p50_ms":
        return d.get("p50_ms")
    if metric_name == "avg_ms":
        return d.get("avg_ms")
    if metric_name == "max_ms":
        return d.get("max_ms")
    return None


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def send_alerts(alerts: list[dict], report: ThresholdReport, event: str, summary: dict) -> list[str]:
    """POST to configured webhooks. Returns list of notified alert types."""
    notified = []
    payload = _alert_payload(report, event, summary)
    for alert in alerts:
        atype = alert.get("type", "generic_webhook")
        if event not in (alert.get("on") or ["threshold_breach", "test_complete"]):
            continue
        url = _resolve_url(alert)
        if not url:
            continue
        try:
            _post_webhook(atype, url, payload, alert)
            notified.append(atype)
        except Exception:
            pass
    return notified


def _resolve_url(alert: dict) -> str | None:
    """Resolve an alert's webhook URL from env or inline config."""
    env_name = alert.get("url_env")
    if env_name:
        return os.environ.get(env_name)
    return alert.get("url")


def _alert_payload(report: ThresholdReport, event: str, summary: dict) -> dict:
    """Build the webhook JSON payload for an alert event."""
    failed = [
        {"rule": r.name, "metric": r.metric, "threshold": r.threshold, "actual": r.actual}
        for r in report.failed
    ]
    return {
        "event": event,
        "text": (
            f"llm-perf {event}: {len(failed)} threshold(s) failed"
            if failed
            else f"llm-perf {event}: all thresholds passed"
        ),
        "failed_rules": failed,
        "summary": summary,
        "ts": time.time(),
    }


def _post_webhook(atype: str, url: str, payload: dict, alert: dict):
    """POST the alert payload to a Slack/PagerDuty/generic endpoint."""
    if atype == "slack":
        failed = payload["failed_rules"]
        lines = (
            "\n".join(f"• `{r['rule']}`: {r['actual']} vs limit {r['threshold']}" for r in failed[:10])
            or "all thresholds passed"
        )
        body = {"text": f":bar_chart: *llm-perf {payload['event']}*\n{lines}"}
    elif atype == "pagerduty":
        routing_key = alert.get("routing_key_env") and os.environ.get(alert["routing_key_env"])
        body = {
            "routing_key": routing_key or alert.get("routing_key", ""),
            "event_action": "trigger" if payload["failed_rules"] else "resolve",
            "payload": {
                "summary": payload["text"],
                "severity": "error" if payload["failed_rules"] else "info",
                "source": "llm-perf",
                "custom_details": {"failed_rules": payload["failed_rules"]},
            },
        }
    else:
        body = payload
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    from .security import check_url, checked_urlopen

    check_url(url)  # webhooks are egress too — same URL policy applies
    checked_urlopen(req, timeout=10).read()
