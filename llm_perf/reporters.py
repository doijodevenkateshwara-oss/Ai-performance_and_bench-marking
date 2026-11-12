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
    "INPUT_TOKENS_PER_SEC": "input_tokens_per_sec",
    "TPOT": "tpot_ms",
    "GEN_DURATION": "gen_duration_ms",
    "ITL_AVG": "itl_avg_ms",
    "ITL_P95": "itl_p95_ms",
    "QUEUE_WAIT": "queue_wait_ms",
    "RETRIES": "retries",
    "TIMEOUTS": "timeouts",
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


def _snap(custom_stats, scenario: str, metric: str) -> dict:
    """Look up exact samples for (scenario, metric); {} when absent."""
    if not custom_stats:
        return {}
    return custom_stats.get((scenario, metric)) or {}


def _enrich_scenario(sc: dict, custom_stats: dict) -> None:
    """Attach derived custom-metric fields to a scenario report entry."""
    name = sc["scenario"]
    ttft_e = _snap(custom_stats, name, "ttft_ms")
    ttft = sc["methods"].get("TTFT", {})
    sc["ttft_p95_ms"] = ttft_e.get("p95") if ttft_e else ttft.get("p95")
    sc["ttft_avg_ms"] = ttft_e.get("avg") if ttft_e else ttft.get("avg_ms")
    toks = sc["methods"].get("TOKENS", {})
    sc["total_tokens"] = int(toks.get("requests", 0) or 0)
    tps_e = _snap(custom_stats, name, "tokens_per_sec")
    tps = sc["methods"].get("TOKENS_PER_SEC", {})
    sc["tokens_per_sec_avg"] = tps_e.get("avg") if tps_e else tps.get("avg_ms")
    itps_e = _snap(custom_stats, name, "input_tokens_per_sec")
    itps = sc["methods"].get("INPUT_TOKENS_PER_SEC", {})
    sc["input_tokens_per_sec_avg"] = itps_e.get("avg") if itps_e else itps.get("avg_ms")
    # latency-family metrics: exact p95 from samples, avg fallback to entry
    for metric, dst, method in (
        ("tpot_ms", "tpot", "TPOT"),
        ("gen_duration_ms", "gen_duration", "GEN_DURATION"),
        ("itl_avg_ms", "itl_avg", "ITL_AVG"),
        ("itl_p95_ms", "itl_p95", "ITL_P95"),
        ("queue_wait_ms", "queue_wait", "QUEUE_WAIT"),
    ):
        se = _snap(custom_stats, name, metric)
        m = sc["methods"].get(method, {})
        sc[f"{dst}_avg_ms"] = se.get("avg") if se else m.get("avg_ms")
        sc[f"{dst}_p95_ms"] = se.get("p95") if se else m.get("p95")
    ret_e = _snap(custom_stats, name, "retries")
    ret = sc["methods"].get("RETRIES", {})
    if ret_e:
        sc["retries_total"] = ret_e.get("sum")
        sc["retries_avg"] = ret_e.get("avg")
    else:
        sc["retries_total"] = (ret.get("avg_ms") or 0.0) * (ret.get("requests") or 0)
        sc["retries_avg"] = ret.get("avg_ms")
    tmo_e = _snap(custom_stats, name, "timeouts")
    tmo = sc["methods"].get("TIMEOUTS", {})
    sc["timeouts"] = int(tmo_e.get("sum") or tmo.get("requests") or 0)
    n = sc.get("requests") or 0
    req_failures = 0
    for m in sc["methods"].values():
        req_failures = max(req_failures, m.get("failures") or 0)
    successes = max(n - req_failures, 1) if n else 0
    cost = sc.get("cost_usd")
    sc["cost_per_success_usd"] = (cost / successes) if cost and successes else None
    _enrich_quality(sc, custom_stats, name)
    _enrich_agent(sc, custom_stats, name)


def _enrich_quality(sc: dict, custom_stats: dict, name: str) -> None:
    q_e = _snap(custom_stats, name, "quality_score")
    q = sc["methods"].get("QUALITY_SCORE", {})
    if q_e:
        sc["quality_score_avg"] = round(q_e.get("avg") or 0.0, 4)
    elif q.get("requests"):
        sc["quality_score_avg"] = round((q.get("avg_ms") or 0) / 1000.0, 4)
    qp_e = _snap(custom_stats, name, "quality_pass")
    if qp_e:
        sc["quality_pass_rate"] = round(qp_e.get("avg") or 0.0, 4)
    cr_e = _snap(custom_stats, name, "context_recall")
    cr = sc["methods"].get("CONTEXT_RECALL", {})
    if cr_e:
        sc["context_recall_avg"] = round(cr_e.get("avg") or 0.0, 4)
    elif cr.get("requests"):
        sc["context_recall_avg"] = round((cr.get("avg_ms") or 0) / 1000.0, 4)


def _enrich_agent(sc: dict, custom_stats: dict, name: str) -> None:
    ag_run = _snap(custom_stats, name, "agent_run")
    if not ag_run:
        return
    sc["agent_runs"] = int(ag_run.get("count") or 0)
    ag_succ = _snap(custom_stats, name, "agent_success")
    sc["agent_success_rate"] = round(ag_succ.get("avg") or 0.0, 4) if ag_succ else None
    for metric, dst in (
        ("agent_steps", "agent_steps_avg"),
        ("agent_tool_calls", "agent_tool_calls_avg"),
        ("agent_tool_errors", "agent_tool_errors_avg"),
        ("agent_loops", "agent_loops_avg"),
    ):
        se = _snap(custom_stats, name, metric)
        sc[dst] = round(se.get("avg"), 3) if se else None
    tc = _snap(custom_stats, name, "agent_tool_calls")
    te = _snap(custom_stats, name, "agent_tool_errors")
    sc["agent_tool_error_rate"] = round((te.get("sum") or 0) / tc["sum"], 4) if tc and tc.get("sum") else None


def aggregate(
    stats,
    config: PerfConfig,
    duration_s: float,
    threshold_report: ThresholdReport | None = None,
    cost_by_scenario: dict | None = None,
    custom_stats: dict | None = None,
) -> dict:
    """Build the full report dict from Locust stats + exact custom samples."""
    cost_by_scenario = cost_by_scenario or {}
    custom_stats = custom_stats or {}
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

    for sc in scenarios.values():
        methods = sc["methods"]
        req = (
            methods.get("")
            or methods.get("POST")
            or methods.get("GET")
            or (max(methods.values(), key=lambda m: m.get("requests", 0)) if methods else {})
            or {}
        )
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
                "cost_usd": cost_by_scenario.get(sc["scenario"]),
            }
        )
        _enrich_scenario(sc, custom_stats)

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
                    "tpot_p95_ms": sc.get("tpot_p95_ms"),
                    "gen_duration_p95_ms": sc.get("gen_duration_p95_ms"),
                    "itl_p95_ms": sc.get("itl_p95_ms"),
                    "queue_wait_p95_ms": sc.get("queue_wait_p95_ms"),
                    "input_tokens_per_sec_avg": sc.get("input_tokens_per_sec_avg"),
                    "retries_total": sc.get("retries_total"),
                    "timeouts": sc.get("timeouts"),
                    "total_tokens": sc.get("total_tokens"),
                    "tokens_per_sec_avg": sc.get("tokens_per_sec_avg"),
                    "quality_score_avg": sc.get("quality_score_avg"),
                    "context_recall_avg": sc.get("context_recall_avg"),
                    "cost_usd": sc.get("cost_usd"),
                    "cost_per_success_usd": sc.get("cost_per_success_usd"),
                    "agent_runs": sc.get("agent_runs"),
                    "agent_success_rate": sc.get("agent_success_rate"),
                    "agent_steps_avg": sc.get("agent_steps_avg"),
                    "agent_tool_calls_avg": sc.get("agent_tool_calls_avg"),
                    "agent_tool_error_rate": sc.get("agent_tool_error_rate"),
                    "agent_loops_avg": sc.get("agent_loops_avg"),
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
