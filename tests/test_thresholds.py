"""Threshold evaluation tests using fake Locust stats entries."""

from types import SimpleNamespace

from llm_perf.config import ThresholdsConfig
from llm_perf.thresholds import evaluate


class FakeEntry:
    def __init__(
        self, name, method, requests=100, failures=0, avg=1000.0, p50=500, p90=1500, p95=2000, p99=3000
    ):
        self.name = name
        self.method = method
        self.num_requests = requests
        self.num_failures = failures
        self.avg_response_time = avg
        self._p = {50: p50, 90: p90, 95: p95, 99: p99}

    def get_response_time_percentile(self, pct):
        return self._p[int(pct)]


class FakeStats:
    def __init__(self, entries):
        self.entries = {f"{e.name}:{e.method}": e for e in entries}
        self.total = SimpleNamespace(
            num_requests=sum(e.num_requests for e in entries),
            num_failures=sum(e.num_failures for e in entries),
        )

    def get(self, name, method):
        return self.entries.get(f"{name}:{method}")


def _stats_with_ttft():
    return FakeStats(
        [
            FakeEntry("chat_load", "", requests=100, failures=1, p95=2000),
            FakeEntry("chat_load", "TTFT", requests=100, avg=300, p50=200, p90=400, p95=500, p99=700),
            FakeEntry("chat_load", "QUALITY_SCORE", requests=20, avg=700.0),  # 0.7
            FakeEntry("chat_load", "CONTEXT_RECALL", requests=10, avg=800.0),
        ]
    )


def test_latency_threshold_pass_and_fail():
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {"chat_load": {"p95_ms": 3000, "ttft_p95_ms": 1000}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    assert rep.all_passed


def test_latency_threshold_breach():
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
    stats = _stats_with_ttft()
    cfg = ThresholdsConfig.model_validate(
        {
            "global": {"max_error_rate": 0.001},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=60, total_requests=100, total_failures=1)
    assert not rep.all_passed  # 1% > 0.1%


def test_skipped_when_no_data():
    stats = FakeStats([])
    cfg = ThresholdsConfig.model_validate(
        {
            "scenarios": {"chat_load": {"p95_ms": 1000}},
        }
    )
    rep = evaluate(stats, cfg, ["chat_load"], duration_s=10, total_requests=0, total_failures=0)
    # no entry -> actual None -> passed None -> all_passed True (skipped)
    assert rep.all_passed
