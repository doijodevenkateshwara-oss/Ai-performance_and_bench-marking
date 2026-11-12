"""SLO threshold evaluation, verdicts, and webhook alerts."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from .config import ThresholdsConfig, ThresholdRule


@dataclass
class RuleResult:
    name: str
    metric: str
    comparator: str          # <= or >=
    threshold: float
    actual: Optional[float]
    passed: Optional[bool]   # None = skipped (no data)
    scope: str = "scenario"  # global | scenario


@dataclass
class ThresholdReport:
    results: list[RuleResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed is not False for r in self.results)

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if r.passed is False]


# metric name -> (stats lookup kind)
_PCT_RULES = [("p50_ms", 50), ("p90_ms", 90), ("p95_ms", 95), ("p99_ms", 99)]
_GAUGE_RULES = [
    ("max_error_rate", "error_rate", "<="),
    ("min_throughput_rps", "throughput_rps", ">="),
    ("min_quality_score", "quality_score", ">="),
    ("min_quality_pass_rate", "quality_pass_rate", ">="),
    ("min_context_recall", "context_recall", ">="),
    ("max_cost_usd", "cost_usd", "<="),
    ("ttft_p95_ms", "ttft_p95", "<="),
]


def _stats_snapshot(stats, name: str):
    """Pull a per-name stats entry (total requests) from Locust stats."""
    try:
        return stats.get(name, name, "GET") if name else None
    except Exception:
        return None


def _entry_for(stats, scenario: str, method: str):
    try:
        return stats.get(scenario, method)
    except Exception:
        return None


def evaluate(stats, thresholds: Optional[ThresholdsConfig], scenario_names: list[str],
             duration_s: float, total_requests: int, total_failures: int,
             cost_by_scenario: Optional[dict] = None,
             stage_stats: Optional[dict] = None) -> ThresholdReport:
    """Evaluate all rules against final Locust stats."""
    rep = ThresholdReport()
    if thresholds is None:
        return rep
    cost_by_scenario = cost_by_scenario or {}
    stage_stats = stage_stats or {}

    def check_rule(rule: ThresholdRule, scope: str, scenario: Optional[str]):
        label = f"{scope}:{scenario}" if scenario else scope
        # latency percentiles — use the scenario's request entry (method "" aggregate)
        entry = _aggregate_entry(stats, scenario) if scenario else _global_entry(stats)
        for field_name, pct in _PCT_RULES:
            limit = getattr(rule, field_name)
            if limit is None:
                continue
            actual = _percentile(entry, pct) if entry else None
            rep.results.append(RuleResult(
                name=f"{label}.{field_name}", metric=f"p{int(pct)}_latency_ms",
                comparator="<=", threshold=limit, actual=actual,
                passed=(actual <= limit) if actual is not None else None, scope=scope))
        # ttft p95
        if rule.ttft_p95_ms is not None:
            e = _entry_for(stats, scenario or "", "TTFT") if scenario else None
            actual = _percentile(e, 95) if e else None
            rep.results.append(RuleResult(
                name=f"{label}.ttft_p95_ms", metric="ttft_p95_ms",
                comparator="<=", threshold=rule.ttft_p95_ms, actual=actual,
                passed=(actual <= rule.ttft_p95_ms) if actual is not None else None,
                scope=scope))
        # gauges
        for field_name, metric, comp in _GAUGE_RULES:
            limit = getattr(rule, field_name)
            if limit is None:
                continue
            actual = _gauge_value(metric, stats, scenario, duration_s,
                                  total_requests, total_failures, cost_by_scenario,
                                  stage_stats)
            if actual is None:
                passed = None
            else:
                passed = (actual <= limit) if comp == "<=" else (actual >= limit)
            rep.results.append(RuleResult(
                name=f"{label}.{field_name}", metric=metric, comparator=comp,
                threshold=limit, actual=actual, passed=passed, scope=scope))
        # per-stage rules
        for stage_name, stage_rule in (rule.stages or {}).items():
            for m_name, limit in stage_rule.items():
                # stage_rule like {"p95_ms": 300}
                actual = _stage_percentile(stage_stats, scenario, stage_name, m_name)
                rep.results.append(RuleResult(
                    name=f"{label}.stage.{stage_name}.{m_name}",
                    metric=f"stage.{stage_name}.{m_name}", comparator="<=",
                    threshold=limit, actual=actual,
                    passed=(actual <= limit) if actual is not None else None,
                    scope=scope))

    if thresholds.global_:
        check_rule(thresholds.global_, "global", None)
    for sc_name, rule in (thresholds.scenarios or {}).items():
        if sc_name in scenario_names or not scenario_names:
            check_rule(rule, "scenario", sc_name)
    return rep


def _aggregate_entry(stats, scenario: str):
    # the request entry is named scenario+method; use aggregate across methods
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
    try:
        return stats.total
    except Exception:
        return None


def _percentile(entry, pct: float) -> Optional[float]:
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


def _gauge_value(metric, stats, scenario, duration_s, total_requests,
                 total_failures, cost_by_scenario, stage_stats):
    try:
        if metric == "error_rate":
            if total_requests <= 0:
                return 0.0
            fails = 0
            if scenario:
                for e in stats.entries.values():
                    if e.name == scenario:
                        fails += e.num_failures
            else:
                fails = total_failures
            return fails / max(total_requests, 1)
        if metric == "throughput_rps":
            return total_requests / max(duration_s, 1e-9)
        if metric == "quality_score":
            e = _entry_for(stats, scenario or "", "QUALITY_SCORE") if scenario else None
            if e is None or e.num_requests == 0:
                return None
            return (e.avg_response_time or 0) / 1000.0
        if metric == "quality_pass_rate":
            p = _entry_for(stats, scenario or "", "QUALITY_SCORE") if scenario else None
            # pass rate approximated: we don't store pass/fail in stats entry; use avg>=0.5
            if p is None or p.num_requests == 0:
                return None
            return 1.0 if (p.avg_response_time or 0) / 1000.0 >= 0.5 else 0.0
        if metric == "context_recall":
            e = _entry_for(stats, scenario or "", "CONTEXT_RECALL") if scenario else None
            if e is None or e.num_requests == 0:
                return None
            return (e.avg_response_time or 0) / 1000.0
        if metric == "cost_usd":
            return cost_by_scenario.get(scenario or "__total__")
    except Exception:
        return None
    return None


def _stage_percentile(stage_stats, scenario, stage_name, metric_name):
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

def send_alerts(alerts: list[dict], report: ThresholdReport, event: str,
                summary: dict) -> list[str]:
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


def _resolve_url(alert: dict) -> Optional[str]:
    env_name = alert.get("url_env")
    if env_name:
        return os.environ.get(env_name)
    return alert.get("url")


def _alert_payload(report: ThresholdReport, event: str, summary: dict) -> dict:
    failed = [{"rule": r.name, "metric": r.metric, "threshold": r.threshold,
               "actual": r.actual} for r in report.failed]
    return {
        "event": event,
        "text": f"llm-perf {event}: {len(failed)} threshold(s) failed" if failed
                else f"llm-perf {event}: all thresholds passed",
        "failed_rules": failed,
        "summary": summary,
        "ts": time.time(),
    }


def _post_webhook(atype: str, url: str, payload: dict, alert: dict):
    if atype == "slack":
        failed = payload["failed_rules"]
        lines = "\n".join(f"• `{r['rule']}`: {r['actual']} vs limit {r['threshold']}"
                          for r in failed[:10]) or "all thresholds passed"
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
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()
