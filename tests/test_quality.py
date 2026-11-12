"""Tests for heuristic checks, LLM-as-judge parsing, and quality scoring.

Author: Venkateshwara Doijode
"""

from llm_perf.config import HeuristicChecks, QualityConfig
from llm_perf.quality import evaluate_quality, run_heuristics
from llm_perf.quality.judge import _parse_score


def test_heuristics_all_pass():
    """All configured checks passing yields score 1.0."""
    cfg = HeuristicChecks(non_empty=True, min_chars=5, max_chars=100, required_patterns=["hello"])
    r = run_heuristics("hello world", cfg)
    assert r.score == 1.0
    assert r.all_passed


def test_heuristics_failures_tagged():
    """Failed checks are tagged in the report."""
    cfg = HeuristicChecks(non_empty=True, forbidden_patterns=["sorry"])
    r = run_heuristics("I am sorry, no", cfg)
    assert not r.all_passed
    assert "forbidden:sorry" in r.failure_tags or "no_refusal" in r.failure_tags


def test_json_valid_check():
    """json_valid check accepts/rejects JSON text."""
    cfg = HeuristicChecks(json_valid=True)
    assert run_heuristics('{"a": 1}', cfg).all_passed
    assert not run_heuristics("not json", cfg).all_passed


def test_refusal_detection():
    """Refusal phrases fail the always-on no_refusal check."""
    cfg = HeuristicChecks()
    r = run_heuristics("I'm sorry, I cannot help with that.", cfg)
    assert "no_refusal" in r.failure_tags


def test_judge_score_parse_json():
    """Judge JSON output parses to a normalized score."""
    score, reason = _parse_score('{"score": 4, "reason": "good"}', 1, 5)
    assert reason == "ok"
    assert abs(score - 0.75) < 1e-6


def test_judge_score_parse_bare_number():
    """Bare numeric judge output parses too."""
    score, _ = _parse_score("Score: 5", 1, 5)
    assert score == 1.0


def test_judge_score_unparseable():
    """Garbage judge output yields None score."""
    score, _reason = _parse_score("I refuse to answer", 1, 5)
    assert score is None


def test_evaluate_quality_combines():
    """Judge score blends 60/40 with heuristic score."""
    qc = QualityConfig(
        heuristics=HeuristicChecks(non_empty=True, min_chars=3),
        judge=None,
    )
    res = evaluate_quality(qc, question="q", response="a good answer")
    assert res.passed is True
    assert res.score == 1.0


def test_evaluate_quality_empty_fails():
    """Empty responses fail non_empty heuristic."""
    qc = QualityConfig(heuristics=HeuristicChecks(non_empty=True))
    res = evaluate_quality(qc, question="q", response="")
    assert res.passed is False
    assert "non_empty" in res.failure_tags
