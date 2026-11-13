"""Tests for the five high-priority measurement fixes.

Covers: per-request latency composition (never percentile subtraction),
IR metrics (Recall@K/Precision@K/MRR/NDCG), quality sampling transparency,
cost-by-call-site breakdown, and distributed shard validation.

Author: Venkateshwara Doijode
"""

import math

from llm_perf.config import ThresholdsConfig
from llm_perf.datasets import validate_shards
from llm_perf.metrics import (
    breakdown_summary,
    cost_stage_snapshot,
    custom_stats_snapshot,
    record_breakdown,
    record_cost,
    record_ir_metrics,
    record_metric,
)
from llm_perf.pipelines.engine import StageContext, _ir_metrics
from llm_perf.thresholds import evaluate

# unique scenario names keep module-level metric stores isolated per test


def _ctx(row):
    return StageContext("q?", row, "ir_test", {})


class TestLatencyComposition:
    """Composition must be computed per request, then aggregated."""

    def test_median_share_is_per_request(self):
        """A stage that is always 50% of e2e reports ~50% share even when
        absolute p95s would misleadingly suggest otherwise."""
        sc = "comp_share"
        # req1: stage=90 of 100; req2: stage=10 of 1000 — shares 90% vs 1%
        record_breakdown(sc, 100.0, {"retrieve": 90.0})
        record_breakdown(sc, 1000.0, {"retrieve": 10.0})
        out = breakdown_summary()[sc]
        seg = out["segments"]["retrieve"]
        # median of per-request shares (0.9, 0.01) -> nearest-rank ~0.9*100
        assert seg["median_share_pct"] == 90.0
        # absolute percentiles still reported for reference
        assert seg["p95_ms"] == 90.0
        # unattributed remainder exists (e2e - stage) per request
        assert "unattributed" in out

    def test_unattributed_remainder(self):
        sc = "comp_unattr"
        record_breakdown(sc, 1000.0, {"generate": 400.0})
        out = breakdown_summary()[sc]
        assert out["unattributed"]["p50_ms"] == 600.0

    def test_empty_and_invalid_ignored(self):
        sc = "comp_empty"
        record_breakdown(sc, 0.0, {"a": 1.0})
        record_breakdown(sc, 100.0, {})
        assert sc not in breakdown_summary()


class TestIrMetrics:
    """Recall@K / Precision@K / MRR / NDCG from relevance judgments."""

    def test_relevant_ids_ordering(self):
        """rel_ids=['d2'], retrieved [d1,d2,d3], k=3 ->
        recall=1/1, precision=1/3, mrr=1/2, ndcg=1/log2(3)."""
        row = {"relevant_ids": ["d2"]}
        docs = [
            {"text": "alpha", "id": "d1"},
            {"text": "beta", "id": "d2"},
            {"text": "gamma", "id": "d3"},
        ]
        m = _ir_metrics(_ctx(row), docs, 3)
        assert m["recall_at_k"] == 1.0
        assert m["precision_at_k"] == pytest_approx(1 / 3)
        assert m["mrr"] == 0.5
        assert m["ndcg"] == pytest_approx(1 / math.log2(3))

    def test_no_judgments_returns_empty(self):
        """Missing ground truth -> {} (metrics absent, not fake zeroes)."""
        docs = [{"text": "a", "id": "d1"}]
        assert _ir_metrics(_ctx({"question": "x"}), docs, 3) == {}
        assert _ir_metrics(_ctx({}), docs, 3) == {}

    def test_relevant_texts_match(self):
        row = {"relevant_texts": ["the quick brown fox"]}
        docs = [
            {"text": "unrelated stuff here"},
            {"text": "well the quick brown fox jumps"},
        ]
        m = _ir_metrics(_ctx(row), docs, 2)
        assert m["recall_at_k"] == 1.0
        assert m["precision_at_k"] == 0.5
        assert m["mrr"] == 0.5

    def test_recorded_as_samples(self):
        record_ir_metrics("ir_rec", {"recall_at_k": 0.75, "mrr": 0.5, "ndcg": None})
        snap = custom_stats_snapshot()
        assert snap[("ir_rec", "recall_at_k")]["avg"] == 0.75
        assert snap[("ir_rec", "mrr")]["avg"] == 0.5
        assert ("ir_rec", "ndcg") not in snap  # None skipped


def pytest_approx(x):
    """Local approx helper (avoids importing pytest.approx at module top)."""
    import pytest

    return pytest.approx(x)


class TestCostBreakdown:
    """Cost accumulated per call site, snapshotted by scenario."""

    def test_stage_snapshot(self):
        sc = "cost_test"
        record_cost(sc, 0.01, stage="generation")
        record_cost(sc, 0.002, stage="reranking")
        record_cost(sc, 0.0005, stage="judge")
        record_cost(sc, 0.01)  # stage=None -> OTel only, not in breakdown
        snap = cost_stage_snapshot()[sc]
        assert snap["generation"] == 0.01
        assert snap["reranking"] == 0.002
        assert snap["judge"] == 0.0005

    def test_zero_cost_ignored(self):
        record_cost("cost_zero", 0.0, stage="generation")
        assert "cost_zero" not in cost_stage_snapshot()


class TestShardValidation:
    """Distributed coverage: union of shards == dataset, no dupes."""

    def test_clean_cover(self):
        v = validate_shards(list(range(10)), 3)
        assert v["ok"] and v["per_worker"] == [4, 3, 3] and v["covered"] == 10

    def test_single_worker(self):
        assert validate_shards([1, 2, 3], 1)["ok"]

    def test_more_workers_than_rows(self):
        v = validate_shards(["a"], 4)
        assert v["ok"] and v["per_worker"] == [1, 0, 0, 0]


class _FakeEntry:
    def __init__(self, name, method, requests=100, avg=0.0):
        self.name = name
        self.method = method
        self.num_requests = requests
        self.num_failures = 0
        self.avg_response_time = avg

    def get_response_time_percentile(self, pct):
        return 0


class _FakeStats:
    def __init__(self, entries):
        self.entries = {f"{e.name}:{e.method}": e for e in entries}

    def get(self, name, method):
        return self.entries.get(f"{name}:{method}")


class TestIrThresholds:
    """New gauge rules evaluate against exact IR samples."""

    def test_min_recall_at_k(self):
        record_metric("th_ir", "recall_at_k", 0.8)
        record_metric("th_ir", "recall_at_k", 0.6)
        cfg = ThresholdsConfig.model_validate(
            {"scenarios": {"th_ir": {"min_recall_at_k": 0.5, "min_ndcg": 0.9}}}
        )
        stats = _FakeStats([_FakeEntry("th_ir", "", requests=10)])
        custom = custom_stats_snapshot()
        rep = evaluate(stats, cfg, ["th_ir"], 60, 10, 0, custom_stats=custom)
        by_name = {r.name: r for r in rep.results}
        assert by_name["scenario:th_ir.min_recall_at_k"].passed is True
        assert by_name["scenario:th_ir.min_recall_at_k"].actual == 0.7
        # no ndcg samples -> not evaluated, not a silent pass/fail
        assert by_name["scenario:th_ir.min_ndcg"].passed is None

    def test_missing_metric_not_evaluated(self):
        cfg = ThresholdsConfig.model_validate({"scenarios": {"th_none": {"min_mrr": 0.5}}})
        stats = _FakeStats([_FakeEntry("th_none", "", requests=10)])
        rep = evaluate(stats, cfg, ["th_none"], 60, 10, 0, custom_stats={})
        assert rep.results[0].passed is None
