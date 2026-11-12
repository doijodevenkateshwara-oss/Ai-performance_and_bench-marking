"""Report aggregation tests with fake Locust stats.

Author: Venkateshwara Doijode
"""

from llm_perf.config import PerfConfig
from llm_perf.reporters import aggregate, write_reports


class FakeEntry:
    """Minimal StatsEntry stand-in for aggregate tests."""

    def __init__(
        self, name, method, requests=50, failures=2, avg=1200.0, p50=900, p90=2000, p95=2500, p99=4000
    ):
        """FakeEntry(num_requests, num_failures, times): canned stats entry."""
        self.name = name
        self.method = method
        self.num_requests = requests
        self.num_failures = failures
        self.avg_response_time = avg
        self.min_response_time = 100
        self.max_response_time = 9000
        self._p = {50: p50, 90: p90, 95: p95, 99: p99}

    def get_response_time_percentile(self, pct):
        """Return a canned percentile."""
        return self._p[int(pct)]


class FakeStats:
    """Minimal RequestStats stand-in (entries + total)."""

    def __init__(self, entries):
        """FakeEntry(num_requests, num_failures, times): canned stats entry."""
        self.entries = {f"{e.name}:{e.method}": e for e in entries}
        self.total = type(
            "T",
            (),
            {
                "num_requests": sum(e.num_requests for e in entries),
                "num_failures": sum(e.num_failures for e in entries),
            },
        )()


def test_aggregate_basic():
    """aggregate() groups entries into per-scenario report rows."""
    stats = FakeStats(
        [
            FakeEntry("chat_load", "", requests=50, failures=2),
            FakeEntry("chat_load", "TTFT", requests=50, avg=400.0, p95=800.0),
            FakeEntry("chat_load", "TOKENS", requests=50, avg=300.0),
            FakeEntry("chat_load", "QUALITY_SCORE", requests=10, avg=750.0),
        ]
    )
    cfg = PerfConfig.model_validate(
        {
            "scenarios": [{"name": "s", "target": "t"}],
            "targets": [{"name": "t", "provider": "openai", "prompts": ["hi"]}],
        }
    )
    report = aggregate(stats, cfg, duration_s=25.0)
    sc = report["scenarios"][0]
    assert sc["scenario"] == "chat_load"
    assert sc["requests"] == 50
    assert sc["failures"] == 2
    assert sc["p95_ms"] == 2500
    assert sc["ttft_p95_ms"] == 800.0
    assert sc["total_tokens"] == 50  # TOKENS entry num_requests
    assert sc["quality_score_avg"] == 0.75
    assert report["totals"]["requests"] == 160  # sum across methods


def test_write_reports_csv_json(tmp_path):
    """CSV and JSON reports are written with expected fields."""
    stats = FakeStats([FakeEntry("s1", "", requests=10)])
    cfg = PerfConfig.model_validate(
        {
            "targets": [{"name": "t", "provider": "openai", "prompts": ["hi"]}],
            "scenarios": [{"name": "s1", "target": "t"}],
            "report": {"formats": ["csv", "json"], "output_dir": str(tmp_path)},
        }
    )
    report = aggregate(stats, cfg, 10.0)
    paths = write_reports(report, cfg)
    assert len(paths) == 2
    assert any(p.suffix == ".csv" for p in paths)
    assert any(p.suffix == ".json" for p in paths)


def test_write_reports_html(tmp_path):
    """HTML report renders via the Jinja template."""
    stats = FakeStats([FakeEntry("s1", "", requests=10)])
    cfg = PerfConfig.model_validate(
        {
            "targets": [{"name": "t", "provider": "openai", "prompts": ["hi"]}],
            "scenarios": [{"name": "s1", "target": "t"}],
            "report": {"formats": ["html"], "output_dir": str(tmp_path)},
        }
    )
    report = aggregate(stats, cfg, 10.0)
    paths = write_reports(report, cfg)
    assert paths[0].suffix == ".html"
    html = paths[0].read_text()
    assert "s1" in html and "p95" in html
