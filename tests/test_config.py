import pytest

from llm_perf.config import PerfConfig


def test_openai_example_validates():
    cfg = PerfConfig.from_yaml("configs/openai_example.yaml")
    assert len(cfg.targets) == 1
    assert cfg.targets[0].provider == "openai"
    assert cfg.targets[0].stream is True
    assert len(cfg.scenarios) == 1
    assert cfg.scenarios[0].quality.judge.rubric


def test_anthropic_example_validates():
    cfg = PerfConfig.from_yaml("configs/anthropic_example.yaml")
    assert cfg.targets[0].provider == "anthropic"


def test_rag_pipeline_validates():
    cfg = PerfConfig.from_yaml("configs/rag_example.yaml")
    t = cfg.targets[0]
    assert t.layer == "pipeline"
    stages = [s.stage for s in t.pipeline]
    assert stages == ["retrieve", "rerank", "generate"]


def test_grpc_example_validates():
    cfg = PerfConfig.from_yaml("configs/grpc_example.yaml")
    assert cfg.targets[0].provider == "grpc"


def test_unknown_target_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
scenarios:
  - name: x
    target: nonexistent
""")
    with pytest.raises(Exception):
        PerfConfig.from_yaml(bad)


def test_dataset_source_resolved_relative(tmp_path):
    (tmp_path / "data.jsonl").write_text('{"prompt": "hi"}\n')
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("""
datasets:
  d:
    source: data.jsonl
    format: jsonl
targets:
  - name: t
    layer: llm_api
    provider: openai
    api_key_env: OPENAI_API_KEY
    prompts: ["hi"]
    dataset: d
scenarios:
  - name: s
    target: t
""")
    cfg = PerfConfig.from_yaml(cfg_file)
    assert cfg.datasets["d"].source.endswith("data.jsonl")
    assert str(tmp_path) in cfg.datasets["d"].source
