"""Post-run analysis: executive summary, bottleneck detection, SLO rollup,
baseline comparison, and regression detection.

All functions operate on the aggregate report dict produced by reporters.aggregate
(+ stages/resources/comparison sections added by the locustfile runner).
"""

from __future__ import annotations

# metrics compared across runs; higher_is_better flips delta interpretation
COMPARE_METRICS = [
    ("error_rate", "error_rate", False),
    ("rps", "requests/s", True),
    ("avg_ms", "avg latency ms", False),
    ("p50_ms", "p50 ms", False),
    ("p90_ms", "p90 ms", False),
    ("p95_ms", "p95 ms", False),
    ("p99_ms", "p99 ms", False),
    ("ttft_p95_ms", "TTFT p95 ms", False),
    ("tpot_p95_ms", "TPOT p95 ms", False),
    ("itl_p95_ms", "ITL p95 ms", False),
    ("gen_duration_p95_ms", "gen duration p95 ms", False),
    ("queue_wait_p95_ms", "queue wait p95 ms", False),
    ("tokens_per_sec_avg", "tokens/s avg", True),
    ("input_tokens_per_sec_avg", "input tokens/s avg", True),
    ("total_tokens", "total tokens", None),
    ("quality_score_avg", "quality score", True),
    ("context_recall_avg", "context recall", True),
    ("retries_total", "retries", False),
    ("timeouts", "timeouts", False),
    ("cost_usd", "cost USD", False),
    ("cost_per_success_usd", "cost/success USD", False),
]

# metrics eligible for regression detection (worse-direction only)
DEFAULT_REGRESSION_THRESHOLDS = {
    "p95_ms": 10.0,
    "p99_ms": 10.0,
    "ttft_p95_ms": 15.0,
    "tpot_p95_ms": 15.0,
    "itl_p95_ms": 15.0,
    "gen_duration_p95_ms": 10.0,
    "queue_wait_p95_ms": 25.0,
    "error_rate": 20.0,  # relative % change allowed on error rate
    "retries_total": 25.0,
    "timeouts": 25.0,
    "cost_usd": 15.0,
    "quality_score_avg": -5.0,  # negative = allowed drop in %
    "tokens_per_sec_avg": -10.0,
    "context_recall_avg": -5.0,
    "rps": -15.0,
}


def _scenarios(report: dict) -> list[dict]:
    return report.get("scenarios") or []


def slo_rollup(report: dict) -> dict:
    """Pass/fail/skip counts for threshold rules."""
    rules = ((report.get("thresholds") or {}).get("rules")) or []
    counts = {"pass": 0, "fail": 0, "skip": 0}
    for r in rules:
        if r.get("passed") is True:
            counts["pass"] += 1
        elif r.get("passed") is False:
            counts["fail"] += 1
        else:
            counts["skip"] += 1
    counts["all_passed"] = counts["fail"] == 0
    return counts


def executive_summary(report: dict) -> dict:
    """Headline numbers + top-line verdict for the HTML/JSON report."""
    totals = report.get("totals") or {}
    scs = _scenarios(report)
    slo = slo_rollup(report)
    reqs = totals.get("requests") or sum(s.get("requests") or 0 for s in scs)
    fails = totals.get("failures") or sum(s.get("failures") or 0 for s in scs)

    # pick the "headline" scenario (highest request count)
    head = max(scs, key=lambda s: s.get("requests") or 0, default=None)

    verdict = "PASS"
    if slo.get("fail"):
        verdict = "FAIL"
    elif (totals.get("error_rate") or 0) > 0.05:
        verdict = "WARN"
    elif not reqs:
        verdict = "NO DATA"

    findings = []
    if head:
        findings.append(
            f"Primary scenario '{head.get('scenario')}' "
            f"served {head.get('requests', 0)} requests "
            f"at p95={_fmt(head.get('p95_ms'))}ms."
        )
    if totals.get("error_rate"):
        findings.append(f"Overall error rate {_pct(totals['error_rate'])}.")
    if head and head.get("tokens_per_sec_avg"):
        findings.append(f"Output throughput ~{head['tokens_per_sec_avg']:.1f} tok/s (avg).")
    cost = sum((s.get("cost_usd") or 0) for s in scs)
    if cost:
        findings.append(f"Estimated cost ${cost:.4f}.")
    if slo.get("fail"):
        findings.append(f"{slo['fail']} SLO rule(s) failed.")
    elif slo.get("pass"):
        findings.append(f"{slo['pass']} SLO rule(s) passed.")

    return {
        "verdict": verdict,
        "requests": reqs,
        "failures": fails,
        "error_rate": totals.get("error_rate"),
        "duration_s": report.get("duration_s"),
        "scenarios": len(scs),
        "headline_scenario": head.get("scenario") if head else None,
        "headline": (
            {
                "p95_ms": head.get("p95_ms") if head else None,
                "ttft_p95_ms": head.get("ttft_p95_ms") if head else None,
                "tokens_per_sec_avg": head.get("tokens_per_sec_avg") if head else None,
                "quality_score_avg": head.get("quality_score_avg") if head else None,
            }
            if head
            else {}
        ),
        "cost_usd": round(cost, 6) or None,
        "slo": slo,
        "findings": findings,
    }


def identify_bottlenecks(report: dict) -> list[dict]:
    """Rank bottlenecks: stage p95 share, high TTFT vs decode split, retries/timeouts."""
    findings: list[dict] = []
    stages = report.get("stages") or []
    scs = _scenarios(report)

    # 1) pipeline stage breakdown — stage with highest p95 share of e2e
    for sc in scs:
        name = sc.get("scenario")
        stages_for = [s for s in stages if s.get("scenario") in (name, None)]
        total_p95 = sc.get("p95_ms") or 0
        for st in sorted(stages_for, key=lambda s: s.get("p95_ms") or 0, reverse=True)[:3]:
            p95 = st.get("p95_ms") or 0
            if p95 <= 0:
                continue
            share = (p95 / total_p95 * 100.0) if total_p95 else 0.0
            if share > 40 or p95 > 5000:
                findings.append(
                    {
                        "type": "stage",
                        "severity": "high" if share > 70 else "medium",
                        "scenario": name,
                        "detail": f"stage '{st.get('stage')}' p95={p95:.0f}ms " f"({share:.0f}% of e2e p95)",
                    }
                )

    # 2) TTFT vs generation split on the headline scenario
    for sc in scs:
        ttft = sc.get("ttft_p95_ms")
        gen = sc.get("gen_duration_p95_ms")
        e2e = sc.get("p95_ms")
        if ttft and gen and e2e:
            if ttft / max(e2e, 1) > 0.6:
                findings.append(
                    {
                        "type": "ttft_bound",
                        "severity": "high" if ttft / e2e > 0.8 else "medium",
                        "scenario": sc.get("scenario"),
                        "detail": f"TTFT p95 ({ttft:.0f}ms) is "
                        f"{ttft / e2e * 100:.0f}% of e2e p95 — prefill/queue-bound",
                    }
                )
            elif gen / max(e2e, 1) > 0.6:
                findings.append(
                    {
                        "type": "decode_bound",
                        "severity": "medium",
                        "scenario": sc.get("scenario"),
                        "detail": f"generation p95 ({gen:.0f}ms) dominates e2e — decode-bound",
                    }
                )

    # 3) retries / timeouts / queue wait
    for sc in scs:
        retries = sc.get("retries_total") or 0
        timeouts = sc.get("timeouts") or 0
        qw = sc.get("queue_wait_p95_ms")
        reqs = sc.get("requests") or 0
        if reqs and (retries / reqs > 0.05 or timeouts / reqs > 0.01):
            findings.append(
                {
                    "type": "reliability",
                    "severity": "high" if (timeouts / reqs) > 0.05 else "medium",
                    "scenario": sc.get("scenario"),
                    "detail": f"{retries} retries, {timeouts} timeouts over {reqs} requests",
                }
            )
        if qw and e2e and (sc.get("p95_ms") or 0) and qw / sc["p95_ms"] > 0.25:
            findings.append(
                {
                    "type": "queueing",
                    "severity": "medium",
                    "scenario": sc.get("scenario"),
                    "detail": f"queue wait p95 {qw:.0f}ms is "
                    f"{qw / sc['p95_ms'] * 100:.0f}% of e2e p95 — load-gen saturation?",
                }
            )

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order.get(f.get("severity"), 3))
    # dedupe near-identical details
    seen, out = set(), []
    for f in findings:
        k = (f["type"], f.get("scenario"), f["detail"][:60])
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out[:20]


def compare_runs(current: dict, baseline: dict | None) -> dict | None:
    """Per-scenario + totals metric deltas vs a baseline report."""
    if not baseline:
        return None
    base_scenarios = {s.get("scenario"): s for s in _scenarios(baseline)}
    cur_scenarios = {s.get("scenario"): s for s in _scenarios(current)}

    def _delta(cur, base, higher_better):
        if cur is None or base in (None, 0):
            return None
        d = cur - base
        pct = (d / base * 100.0) if base else None
        direction = None
        if pct is not None and higher_better is not None:
            direction = "better" if (d > 0) == higher_better else ("worse" if d else "same")
        return {"current": cur, "baseline": base, "delta": d, "delta_pct": pct, "direction": direction}

    metrics = []
    for key, label, hb in COMPARE_METRICS:
        # aggregate across scenarios (sum for counts, request-weighted for rates)
        cur_val = _agg_metric(cur_scenarios.values(), key)
        base_val = _agg_metric(base_scenarios.values(), key)
        metrics.append(
            {
                "metric": key,
                "label": label,
                "delta": _delta(cur_val, base_val, hb),
            }
        )

    per_scenario = []
    for name in sorted(set(cur_scenarios) | set(base_scenarios)):
        cur, base = cur_scenarios.get(name), base_scenarios.get(name)
        if not cur or not base:
            per_scenario.append(
                {"scenario": name, "note": "missing in " + ("baseline" if not base else "current")}
            )
            continue
        diffs = {}
        for key, label, hb in COMPARE_METRICS:
            d = _delta(cur.get(key), base.get(key), hb)
            if d and d["delta_pct"] is not None:
                diffs[key] = {"label": label, **d}
        per_scenario.append({"scenario": name, "metrics": diffs})

    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "baseline_profile": (baseline.get("load") or {}).get("profile"),
        "metrics": metrics,
        "per_scenario": per_scenario,
    }


def _agg_metric(scenarios, key: str):
    vals = [s.get(key) for s in scenarios if s.get(key) is not None]
    if not vals:
        return None
    if key in ("total_tokens", "retries_total", "timeouts", "cost_usd"):
        return sum(vals)
    # rates/latencies: request-weighted average
    num = den = 0.0
    for s in scenarios:
        v = s.get(key)
        w = s.get("requests") or 0
        if v is not None and w:
            num += v * w
            den += w
    return (num / den) if den else (sum(vals) / len(vals))


def detect_regressions(
    current: dict,
    baseline: dict | None,
    thresholds_pct: dict | None = None,
) -> dict:
    """Flag metrics that got worse vs baseline beyond allowed % change."""
    if not baseline:
        return {"checked": False, "regressions": []}
    limits = dict(DEFAULT_REGRESSION_THRESHOLDS)
    limits.update(thresholds_pct or {})

    comp = compare_runs(current, baseline)
    regressions = []
    for m in (comp or {}).get("metrics", []):
        key = m["metric"]
        d = m.get("delta")
        if not d or d["delta_pct"] is None:
            continue
        allowed = limits.get(key)
        if allowed is None:
            continue
        pct = d["delta_pct"]
        # negative allowed = metric should not drop more than |allowed|%
        breached = pct < allowed if allowed < 0 else pct > allowed  # e.g. -10 = drop >10% bad
        if breached:
            regressions.append(
                {
                    "metric": key,
                    "label": m["label"],
                    "current": d["current"],
                    "baseline": d["baseline"],
                    "delta_pct": pct,
                    "allowed_pct": allowed,
                }
            )
    # per-scenario regressions
    for ps in (comp or {}).get("per_scenario", []):
        for key, d in (ps.get("metrics") or {}).items():
            allowed = limits.get(key)
            if allowed is None or d.get("delta_pct") is None:
                continue
            pct = d["delta_pct"]
            breached = pct < allowed if allowed < 0 else pct > allowed
            if breached:
                regressions.append(
                    {
                        "metric": key,
                        "label": d["label"],
                        "scenario": ps.get("scenario"),
                        "current": d["current"],
                        "baseline": d["baseline"],
                        "delta_pct": pct,
                        "allowed_pct": allowed,
                    }
                )
    return {"checked": True, "regressions": regressions}


def _fmt(v) -> str:
    return f"{v:.0f}" if isinstance(v, (int, float)) else "n/a"


def _pct(v) -> str:
    return f"{v * 100:.2f}%" if isinstance(v, (int, float)) else "n/a"
