"""Startup env-var validation: missing api_key_env fails loud.

Author: Venkateshwara Doijode
"""

import pytest
import typer
import yaml

from llm_perf.cli import _check_env_vars
from llm_perf.config import PerfConfig

_BASE = {
    "targets": [
        {
            "name": "t1",
            "layer": "llm_api",
            "provider": "openai",
            "api_key_env": "LLMPERF_TEST_MISSING_KEY",
            "params": {"model": "gpt-4o-mini"},
        }
    ],
    "scenarios": [{"name": "s1", "target": "t1", "prompts": ["hi"]}],
    "datasets": {},
}


def _cfg(tmp_path, monkeypatch, extra_target=None):
    data = dict(_BASE)
    if extra_target:
        data["targets"] = data["targets"] + [extra_target]
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    return PerfConfig.from_yaml(str(p))


def test_missing_api_key_env_fails_loud(tmp_path, monkeypatch):
    """Missing api_key_env exits nonzero at startup, not mid-run."""
    monkeypatch.delenv("LLMPERF_TEST_MISSING_KEY", raising=False)
    cfg = _cfg(tmp_path, monkeypatch)
    with pytest.raises(typer.Exit) as exc:
        _check_env_vars(cfg)
    assert exc.value.exit_code == 2


def test_present_api_key_env_passes(tmp_path, monkeypatch):
    """Set env var -> validation passes."""
    monkeypatch.setenv("LLMPERF_TEST_MISSING_KEY", "sk-test")
    cfg = _cfg(tmp_path, monkeypatch)
    _check_env_vars(cfg)  # no raise


def test_nested_pipeline_llm_key_checked(tmp_path, monkeypatch):
    """api_key_env inside a pipeline stage llm: block is also validated."""
    cfg = _cfg(
        tmp_path,
        monkeypatch,
        extra_target={
            "name": "rag",
            "layer": "pipeline",
            "dataset": "d",
            "pipeline": [
                {"stage": "generate", "llm": {"provider": "openai", "api_key_env": "LLMPERF_TEST_NESTED_KEY"}}
            ],
        },
    )
    monkeypatch.delenv("LLMPERF_TEST_NESTED_KEY", raising=False)
    with pytest.raises(typer.Exit):
        _check_env_vars(cfg)


def test_missing_dataset_source_warns_not_fails(tmp_path, monkeypatch):
    """Missing dataset file warns but does not block the run."""
    monkeypatch.setenv("LLMPERF_TEST_MISSING_KEY", "sk-test")
    data = dict(_BASE)
    data["datasets"] = {"d": {"source": "/nonexistent/file.jsonl"}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    cfg = PerfConfig.from_yaml(str(p))
    _check_env_vars(cfg)  # warning only
