"""Tests for benchmark-driven optimization candidates and scoring.

Author: Venkateshwara Doijode
"""

import json
from pathlib import Path

import pytest
import yaml

from llm_perf import analysis
from llm_perf.config import PerfConfig, ThresholdRule
from llm_perf.optimizer import candidate_configs, run_optimization, score_report


def _config(tmp_path: Path, optimization: dict | None = None) -> Path:
    """Write a small valid pipeline config."""
    data = tmp_path / "qa.jsonl"
    docs = tmp_path / "docs.jsonl"
    data.write_text('{"question": "q?", "expected_answer": "a"}\n')
    docs.write_text('{"id": "d1", "text": "answer context"}\n')
    cfg = {
        "observability": {"otel": {"enabled": False}},
        "datasets": {"qa": {"source": str(data), "format": "jsonl"}},
        "targets": [
            {
                "name": "rag",
                "layer": "pipeline",
                "dataset": "qa",
                "pipeline": [
                    {
                        "stage": "retrieve",
                        "vector_store": {"type": "json", "path": str(docs)},
                        "top_k": 5,
                    },
                    {
                        "stage": "generate",
                        "llm": {
                            "provider": "openai",
                            "model": "mock-model",
                            "params": {"max_tokens": 256},
                        },
                    },
                ],
            }
        ],
        "scenarios": [{"name": "rag_load", "target": "rag"}],
        "load": {"users": 1, "spawn_rate": 1, "duration": "1s"},
        "report": {"formats": ["json"], "output_dir": str(tmp_path / "results")},
        "optimization": optimization
        or {
            "enabled": True,
            "scenario": "rag_load",
            "objective": "p95_ms",
            "parameters": [
                {
                    "target": "rag",
                    "stage_index": 0,
                    "path": "top_k",
                    "values": [3],
                },
                {
                    "target": "rag",
                    "stage_index": 1,
                    "path": "llm.params.max_tokens",
                    "values": [128],
                },
            ],
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_optimization_config_parses(tmp_path):
    """optimization is part of the root config schema."""
    cfg = PerfConfig.from_yaml(_config(tmp_path))
    assert cfg.optimization.enabled is True
    assert cfg.optimization.objective == "p95_ms"
    assert cfg.optimization.parameters[0].path == "top_k"


def test_candidate_configs_include_baseline_and_mutation(tmp_path):
    """Grid generation emits baseline plus the requested candidate."""
    cfg = PerfConfig.from_yaml(_config(tmp_path))
    candidates = candidate_configs(cfg)
    assert len(candidates) == 2
    assert candidates[0].label == "baseline"
    trial = candidates[1].config.targets[0]
    assert trial.pipeline[0].top_k == 3
    assert trial.pipeline[1].llm["params"]["max_tokens"] == 128
    assert candidates[1].values["rag.pipeline[0].top_k"] == 3


def test_candidate_generation_rejects_unsafe_stage_path(tmp_path):
    """Credentials/endpoints are not optimization-tunable."""
    path = _config(
        tmp_path,
        {
            "enabled": True,
            "scenario": "rag_load",
            "parameters": [
                {
                    "target": "rag",
                    "stage_index": 1,
                    "path": "llm.api_key_env",
                    "values": ["SECRET"],
                }
            ],
        },
    )
    cfg = PerfConfig.from_yaml(path)
    with pytest.raises(ValueError, match="unsupported pipeline optimization path"):
        candidate_configs(cfg)


def test_candidate_generation_rejects_missing_target(tmp_path):
    """A parameter must reference a configured target."""
    path = _config(
        tmp_path,
        {
            "enabled": True,
            "scenario": "rag_load",
            "parameters": [{"target": "missing", "path": "params.model", "values": ["x"]}],
        },
    )
    cfg = PerfConfig.from_yaml(path)
    with pytest.raises(ValueError, match="target not found"):
        candidate_configs(cfg)


def _report(p95=800.0, quality=0.8, error_rate=0.01):
    """Build a minimal optimizer score report."""
    return {
        "scenarios": [
            {
                "scenario": "rag_load",
                "requests": 50,
                "error_rate": error_rate,
                "p95_ms": p95,
                "quality_score_avg": quality,
                "agent_success_rate": 0.9,
                "cost_per_success_usd": 0.01,
            }
        ],
        "stages": [
            {
                "scenario": "rag_load",
                "stage": "retrieve",
                "p95_ms": 100.0,
            }
        ],
    }


def test_score_report_applies_constraints():
    """Feasibility requires latency, quality, and error constraints."""
    rule = ThresholdRule.model_validate(
        {
            "p95_ms": 1000,
            "max_error_rate": 0.02,
            "min_quality_score": 0.7,
            "stages": {"retrieve": {"p95_ms": 150}},
        }
    )
    scored = score_report(_report(), "rag_load", "p95_ms", constraints=rule, min_requests=10)
    assert scored["feasible"] is True
    assert scored["score"] == 800.0

    bad = score_report(
        _report(quality=0.4, error_rate=0.1),
        "rag_load",
        "p95_ms",
        constraints=rule,
        min_requests=10,
    )
    assert bad["feasible"] is False
    assert any("max_error_rate" in v for v in bad["violations"])
    assert any("min_quality_score" in v for v in bad["violations"])


def test_score_report_requires_enough_requests():
    """A tiny/no-sample trial cannot win by accident."""
    scored = score_report(_report(), "rag_load", "p95_ms", min_requests=100)
    assert scored["feasible"] is False
    assert scored["violations"] == ["insufficient requests: 50 < 100"]


def test_run_optimization_selects_winner_and_writes_artifacts(tmp_path):
    """A fake runner proves trial orchestration and optimized.yaml output."""
    cfg_path = _config(tmp_path)
    out_dir = tmp_path / "opt"

    def fake_runner(candidate_path: Path) -> int:
        data = yaml.safe_load(candidate_path.read_text())
        out = Path(data["report"]["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        p95 = 900.0 if candidate_path.name == "trial-00.yaml" else 500.0
        report = _report(p95=p95)
        (out / f"llm-perf-{candidate_path.stem}.json").write_text(json.dumps(report))
        return 0

    result = run_optimization(cfg_path, output_dir=out_dir, runner=fake_runner)

    assert result["winner"]["trial"] == 1
    assert result["winner"]["score"] == 500.0
    assert Path(result["report_path"]).exists()
    optimized = Path(result["optimized_config"])
    assert optimized.exists()
    optimized_cfg = PerfConfig.from_yaml(optimized)
    assert optimized_cfg.targets[0].pipeline[0].top_k == 3
    assert optimized_cfg.targets[0].pipeline[1].llm["params"]["max_tokens"] == 128
    assert optimized_cfg.report.output_dir == str(tmp_path / "results")
    assert optimized_cfg.optimization.enabled is False


def test_optimization_recommendations_are_stage_aware():
    """Dominant stage findings map to structured tuning guidance."""
    report = {
        "scenarios": [{"scenario": "rag_load", "p95_ms": 1000.0}],
        "composition": {
            "rag_load": {
                "segments": {
                    "generate": {"median_share_pct": 80.0, "p95_ms": 900.0},
                    "retrieve": {"median_share_pct": 10.0, "p95_ms": 80.0},
                }
            }
        },
    }
    recs = analysis.optimization_recommendations(report)
    assert recs
    assert recs[0]["action"] == "reduce_generation_latency"
    assert recs[0]["stage"] == "generate"
    assert "llm.params.max_tokens" in recs[0]["parameters"]
