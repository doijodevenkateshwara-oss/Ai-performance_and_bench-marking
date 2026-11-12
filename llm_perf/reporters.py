"""End-of-run statistical reports: CSV / JSON / HTML."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import PerfConfig
from .thresholds import ThresholdReport

_METHOD_LABELS = {
    "": "requests",
    "GET": "requests",
    "POST": "requests",
    "TTFT": "ttft_ms",
    "TOKENS": "tokens",
    "PROMPT_TOKENS": "prompt_tokens",
    "COMPLETION_TOKENS": "completion_tokens",
    "TOKENS_PER_SEC": "tokens_per_sec",
    "QUALITY_SCORE": "quality_score_x1000",
    "CONTEXT_RECALL": "context_recall_x1000",
}


def _entry_stats(entry) -> dict:
    if entry is None:
        return {}
    try:
        return {
            "requests": entry.num_requests,
            "failures": entry.num_failures,
            "error_rate": (entry.num_failures / entry.num_requests) if entry.num_requests else 0.0,
            "avg_ms": entry.avg_response_time,
            "min_ms": entry.min_response_time,
            "max_ms": entry.max_response_time,
            "p50": entry.get_response_time_percentile(50),
            "p90": entry.get_response_time_percentile(90),
            "p95": entry.get_response_time_percentile(95),
            "p99": entry.get_response_time_percentile(99),
        }
    except Exception:
        return {}


def aggregate(
    stats,
    config: PerfConfig,
    duration_s: float,
    threshold_report: ThresholdReport | None = None,
    cost_by_scenario: dict | None = None,
) -> dict:
    """Build the full report dict from Locust stats."""
    cost_by_scenario = cost_by_scenario or {}
    scenarios: dict[str, dict] = {}
    # group entries by scenario name
    try:
        entries = list(stats.entries.values())
    except Exception:
        entries = []
    for e in entries:
        name = e.name
        method = e.method
        sc = scenarios.setdefault(name, {"scenario": name, "methods": {}})
        sc["methods"][method or ""] = _entry_stats(e)

    # totals
    total_req = (
        getattr(stats, "total_num_requests", lambda: 0)()
        if callable(getattr(stats, "total_num_requests", None))
        else getattr(stats, "num_requests", 0)
    )
    try:
        total_req = stats.total.num_requests
        total_fail = stats.total.num_failures
    except Exception:
        total_req, total_fail = 0, 0

    for name, sc in scenarios.items():
        req = sc["methods"].get("", {}) or sc["methods"].get("GET", {}) or {}
        n = req.get("requests", 0)
        sc.update(
            {
                "requests": n,
                "failures": req.get("failures", 0),
                "error_rate": req.get("error_rate", 0.0),
                "avg_ms": req.get("avg_ms"),
                "p50_ms": req.get("p50"),
                "p90_ms": req.get("p90"),
                "p95_ms": req.get("p95"),
                "p99_ms": req.get("p99"),
                "rps": (n / duration_s) if duration_s else None,
                "cost_usd": cost_by_scenario.get(name),
            }
        )
        # derived custom stats
        ttft = sc["methods"].get("TTFT", {})
        sc["ttft_p95_ms"] = ttft.get("p95")
        sc["ttft_avg_ms"] = ttft.get("avg_ms")
        toks = sc["methods"].get("TOKENS", {})
        sc["total_tokens"] = int(toks.get("requests", 0) or 0)
        tps = sc["methods"].get("TOKENS_PER_SEC", {})
        sc["tokens_per_sec_avg"] = tps.get("avg_ms")
        q = sc["methods"].get("QUALITY_SCORE", {})
        if q.get("requests"):
            sc["quality_score_avg"] = round((q.get("avg_ms") or 0) / 1000.0, 4)
        cr = sc["methods"].get("CONTEXT_RECALL", {})
        if cr.get("requests"):
            sc["context_recall_avg"] = round((cr.get("avg_ms") or 0) / 1000.0, 4)

    verdicts = []
    if threshold_report:
        for r in threshold_report.results:
            verdicts.append(
                {
                    "rule": r.name,
                    "metric": r.metric,
                    "comparator": r.comparator,
                    "threshold": r.threshold,
                    "actual": r.actual,
                    "passed": r.passed,
                    "scope": r.scope,
                }
            )

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": duration_s,
        "totals": {
            "requests": total_req,
            "failures": total_fail,
            "error_rate": (total_fail / total_req) if total_req else 0.0,
        },
        "scenarios": list(scenarios.values()),
        "thresholds": {
            "all_passed": threshold_report.all_passed if threshold_report else None,
            "failed": [r.name for r in threshold_report.failed] if threshold_report else [],
            "rules": verdicts,
        },
    }


def write_reports(
    report: dict, config: PerfConfig, threshold_report: ThresholdReport | None = None
) -> list[Path]:
    out_dir = Path(config.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    written = []

    if "json" in config.report.formats:
        p = out_dir / f"llm-perf-{ts}.json"
        p.write_text(json.dumps(report, indent=2, default=str))
        written.append(p)

    if "csv" in config.report.formats:
        p = out_dir / f"llm-perf-{ts}.csv"
        rows = []
        for sc in report["scenarios"]:
            rows.append(
                {
                    "scenario": sc["scenario"],
                    "requests": sc.get("requests"),
                    "failures": sc.get("failures"),
                    "error_rate": sc.get("error_rate"),
                    "avg_ms": sc.get("avg_ms"),
                    "p50_ms": sc.get("p50_ms"),
                    "p90_ms": sc.get("p90_ms"),
                    "p95_ms": sc.get("p95_ms"),
                    "p99_ms": sc.get("p99_ms"),
                    "rps": sc.get("rps"),
                    "ttft_p95_ms": sc.get("ttft_p95_ms"),
                    "total_tokens": sc.get("total_tokens"),
                    "tokens_per_sec_avg": sc.get("tokens_per_sec_avg"),
                    "quality_score_avg": sc.get("quality_score_avg"),
                    "context_recall_avg": sc.get("context_recall_avg"),
                    "cost_usd": sc.get("cost_usd"),
                }
            )
        if rows:
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        written.append(p)

    if "html" in config.report.formats:
        p = out_dir / f"llm-perf-{ts}.html"
        env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            autoescape=select_autoescape(["html"]),
        )
        tpl = env.get_template("report.html")
        p.write_text(tpl.render(report=report))
        written.append(p)

    return written
