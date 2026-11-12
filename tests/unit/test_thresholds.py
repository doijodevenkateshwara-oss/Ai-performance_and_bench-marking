"""Threshold evaluation tests using fake Locust stats entries.

Author: Venkateshwara Doijode
"""

from types import SimpleNamespace

from llm_perf.config import ThresholdsConfig
from llm_perf.thresholds import evaluate


class FakeEntry:
    """Minimal StatsEntry stand-in for threshold tests."""

    def __init__(
        self, name, method, requests=100, failures=0, avg=1000.0, p50=500, p90=1500, p95=2000, p99=3000
    ):
        """FakeEntry(num_requests, num_failures, times): canned stats entry."""
        self.name = name
        self.method = method
        self.num_requests = requests
        self.num_failures = failures
        self.avg_response_time = avg
        self._p = {50: p50, 90: p90, 95: p95, 99: p99}

    def get_response_time_percentile(self, pct):
        """Return a canned percentile."""
        return self._p[int(pct)]


class FakeStats:
    """Minimal RequestStats stand-in (entries + total)."""

    def __init__(self, entries):
        """FakeEntry(num_requests, num_failures, times): canned stats entry."""
        self.entries = {f"{e.name}:{e.method}": e for e in entries}
        self.total = SimpleNamespace(
            num_requests=sum(e.num_requests for e in entries),
            num_failures=sum(e.num_failures for e in entries),
        )

    def get(self, name, method):
        """Fetch an entry by (name, method)."""
        return self.entries.get(f"{name}:{method}")


def _stats_with_ttft():
    """Build stats containing request + TTFT custom entries."""
    return FakeStats(
        [
            FakeEntry("chat_load", "", requests=100, failures=1, p95=2000),
            FakeEntry("chat_load", "TTFT", requests=100, avg=300, p50=200, p90=400, p95=500, p99=700),
            FakeEntry("chat_load", "QUALITY_SCORE", requests=20, avg=700.0),  # 0.7
            FakeEntry("chat_load", "CONTEXT_RECALL", requests=10, avg=800.0),
        ]
    )


def test_latency_threshold_pass_and_fail():
    """p95 rules pass/fail against entry percentiles."""
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {"chat_load": {"p95_ms": 3000, "ttft_p95_ms": 1000}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    assert rep.all_passed


def test_latency_threshold_breach():
    """A breached p95 rule reports FAIL with actual value."""
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {"chat_load": {"p95_ms": 1000}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    assert not rep.all_passed
    failed = rep.failed[0]
    assert failed.actual == 2000 and failed.threshold == 1000


def test_error_rate_and_throughput():
    """error_rate and rps gauges evaluate correctly."""
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {"chat_load": {"max_error_rate": 0.5, "min_throughput_rps": 0.5}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    # error rate computed per-scenario from entries named chat_load
    assert rep.all_passed


def test_quality_score_threshold():
    """quality_score gauge uses exact samples when present."""
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {"chat_load": {"min_quality_score": 0.8}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    # quality avg = 700/1000 = 0.7 < 0.8 -> fail
    q = [r for r in rep.results if "quality" in r.name]
    assert q and q[0].passed is False


def test_global_rule_applies():
    """global-scope rules evaluate once without a scenario."""
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "global": {"max_error_rate": 0.001},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    assert not rep.all_passed  # 1% > 0.1%


def test_skipped_when_no_data():
    """Rules with no data report passed=None (skipped)."""
    stats = FakeStats([])
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {"chat_load": {"p95_ms": 1000}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=10, total_requests=0, total_failures=0)
    # no entry -> actual None -> passed None -> all_passed True (skipped)
    assert rep.all_passed
    rule = rep.results[0]
    assert rule.status == "not_evaluated"


def test_not_evaluated_policy_pass_default():
    """Default on_not_evaluated=pass: unevaluated rules don't fail CI."""
    stats = FakeStats([])
    cfg = ThresholdsConfig.model_validate({"scenarios": {"chat_load": {"p95_ms": 1000}}})
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=10, total_requests=0, total_failures=0)
    assert rep.not_evaluated and not rep.should_fail_ci and not rep.should_warn


def test_not_evaluated_policy_fail():
    """on_not_evaluated=fail makes missing data a CI failure."""
    stats = FakeStats([])
    cfg = ThresholdsConfig.model_validate(
        {
            "on_not_evaluated": "fail",
            "scenarios": {"chat_load": {"p95_ms": 1000}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=10, total_requests=0, total_failures=0)
    assert rep.should_fail_ci and not rep.should_warn


def test_not_evaluated_policy_warn():
    """on_not_evaluated=warn surfaces a warning without failing CI."""
    stats = FakeStats([])
    cfg = ThresholdsConfig.model_validate(
        {
            "on_not_evaluated": "warn",
            "scenarios": {"chat_load": {"p95_ms": 1000}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=10, total_requests=0, total_failures=0)
    assert rep.should_warn and not rep.should_fail_ci


def test_status_three_states():
    """RuleResult.status maps passed True/False/None to pass/fail/not_evaluated."""
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {
                "chat_load": {
                    "p95_ms": 3000,  # pass (actual 2000)
                    "ttft_p95_ms": 100,  # fail (actual 500)
                    "itl_p95_ms": 50,  # no TPOT/ITL data -> not_evaluated
                },
            },
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    states = {r.status for r in rep.results}
    assert {"pass", "fail", "not_evaluated"} <= states
