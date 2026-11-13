"""Reproducibility manifest tests.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json

from llm_perf import manifest
from llm_perf.config import PerfConfig


def _config(tmp_path) -> PerfConfig:
    """Minimal valid config exercising targets, datasets, pricing."""
    ds = tmp_path / "rows.jsonl"
    ds.write_text('{"question": "q1", "answer": "a1"}\n')
    cfg = PerfConfig.model_validate(
        {
            "datasets": {"qa": {"source": str(ds), "format": "jsonl"}},
            "targets": [
                {
                    "name": "chat",
                    "layer": "llm_api",
                    "provider": "openai",
                    "base_url": "https://user:secret@example.com:8443/v1",
                    "params": {"model": "gpt-4o", "api_key": "sk-live-123"},
                    "aws": {"region": "us-east-1"},
                }
            ],
            "load": {"users": 2, "spawn_rate": 2, "duration": "10s"},
            "report": {"pricing": {"input_per_1k": 0.001, "output_per_1k": 0.002}},
        }
    )
    return cfg


def test_manifest_shape(tmp_path):
    """Manifest contains all provenance sections with expected keys."""
    man = manifest.build(_config(tmp_path), "cfg.yaml")
    for key in (
        "run_id",
        "generated_at",
        "framework",
        "git",
        "config",
        "datasets",
        "targets",
        "load",
        "load_generator",
        "pricing",
    ):
        assert key in man, key
    assert man["config"]["path"] == "cfg.yaml"
    assert len(man["config"]["sha256"]) == 64
    assert man["framework"]["name"] == "llm-perf"
    assert man["load"]["users"] == 2
    assert man["pricing"]["input_per_1k"] == 0.001


def test_manifest_dataset_hash(tmp_path):
    """Dataset entries carry raw-byte sha256 + source + size."""
    man = manifest.build(_config(tmp_path))
    entry = man["datasets"]["qa"]
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert entry["bytes"] == len(b'{"question": "q1", "answer": "a1"}\n')
    assert entry["source"].endswith("rows.jsonl")


def test_manifest_target_redaction_and_host(tmp_path):
    """Targets expose host (no userinfo) and provider/model; secrets stay out."""
    man = manifest.build(_config(tmp_path))
    t = man["targets"][0]
    assert t["provider"] == "openai"
    assert t["model"] == "gpt-4o"
    assert t["host"] == "example.com:8443"  # port kept, userinfo dropped
    assert t["region"] == "us-east-1"
    blob = json.dumps(man)
    assert "secret@example.com" not in blob
    assert "sk-live-123" not in blob


def test_config_hash_stable_and_sensitive_invariant(tmp_path):
    """Same config -> same hash; secret values don't change the hash."""
    cfg1 = _config(tmp_path)
    cfg2 = PerfConfig.model_validate(
        {
            "datasets": {"qa": {"source": str(tmp_path / "rows.jsonl"), "format": "jsonl"}},
            "targets": [
                {
                    "name": "chat",
                    "layer": "llm_api",
                    "provider": "openai",
                    "base_url": "https://user:secret@example.com:8443/v1",
                    "params": {"model": "gpt-4o", "api_key": "sk-DIFFERENT"},
                    "aws": {"region": "us-east-1"},
                }
            ],
            "load": {"users": 2, "spawn_rate": 2, "duration": "10s"},
            "report": {"pricing": {"input_per_1k": 0.001, "output_per_1k": 0.002}},
        }
    )
    # api_key redacted -> identical canonical payload -> identical hash
    assert manifest._config_hash(cfg1) == manifest._config_hash(cfg2)


def test_git_info_never_raises(tmp_path, monkeypatch):
    """git probing degrades to Nones outside a repo / on failure."""
    monkeypatch.setattr(manifest, "_git_info", lambda: {"commit": None, "dirty": None, "root": None})
    man = manifest.build(_config(tmp_path))
    assert man["git"]["commit"] is None


def test_provenance_drift_detection():
    """compare_runs flags config/dataset/code drift vs baseline."""
    from llm_perf import analysis

    cur = {
        "manifest": {
            "config": {"sha256": "aaa"},
            "datasets": {"qa": {"sha256": "d1"}},
            "git": {"commit": "c1"},
            "framework": {"version": "0.1.0"},
        }
    }
    base = {
        "manifest": {
            "config": {"sha256": "bbb"},
            "datasets": {"qa": {"sha256": "d1"}},
            "git": {"commit": "c1"},
            "framework": {"version": "0.1.0"},
        },
        "generated_at": "t",
        "load": {"profile": "smoke"},
        "scenarios": [],
    }
    cur["scenarios"] = []
    out = analysis.compare_runs(cur, base)
    assert out["provenance"]["comparable"] is False
    fields = {d["field"] for d in out["provenance"]["drift"]}
    assert fields == {"config_sha256"}

    same = dict(cur)
    same["manifest"] = json.loads(json.dumps(base["manifest"]))
    out2 = analysis.compare_runs(same, base)
    assert out2["provenance"]["comparable"] is True
