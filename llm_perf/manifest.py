"""Reproducibility manifest — provenance block embedded in every report.

Captures everything needed to explain *why* two runs differ: framework and
dependency versions, git state, canonical config + dataset hashes, per-target
provider/model identity, environment, and the pricing snapshot used for cost
estimates. All fields are best-effort — provenance must never fail a run, so
every collector degrades to ``None`` on error.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .config import AppEndpointTarget, AppObjectTarget, LlmApiTarget, PerfConfig, PipelineTarget

# keys whose values may contain credentials — redacted before hashing/dumping
_SENSITIVE_KEY = re.compile(r"key|token|secret|password|credential|authorization", re.IGNORECASE)

_REDACTED = "<redacted>"


def _redact(obj):
    """Recursively replace values under sensitive-looking keys."""
    if isinstance(obj, dict):
        return {k: (_REDACTED if _SENSITIVE_KEY.search(str(k)) else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


def _sha256(data: bytes | str) -> str:
    """Hex SHA-256 of bytes or text."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _dist_version(dist: str) -> str | None:
    """Installed distribution version, or None when unavailable."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_info() -> dict:
    """Best-effort git commit + dirty flag for the repo containing this package."""
    repo = Path(__file__).resolve().parent.parent
    info: dict = {"commit": None, "dirty": None, "root": None}
    git = shutil.which("git")
    if git is None:
        return info
    try:
        commit = subprocess.run(  # noqa: S603 - fixed argv, resolved git binary, no shell
            [git, "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip() or None
        status = subprocess.run(  # noqa: S603 - fixed argv, resolved git binary, no shell
            [git, "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
        info["root"] = str(repo) if info["commit"] else None
    except Exception:
        pass
    return info


def _host(url: str | None) -> str | None:
    """Hostname[:port] of a URL — never includes userinfo/path/query."""
    if not url:
        return None
    try:
        p = urlparse(url)
        if not p.hostname:
            return None
        return f"{p.hostname}:{p.port}" if p.port else p.hostname
    except Exception:
        return None


def _config_hash(config: PerfConfig) -> str:
    """SHA-256 of the canonical (sorted, secret-redacted) effective config."""
    payload = _redact(config.model_dump(mode="json", by_alias=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256(canonical)


def _dataset_entries(config: PerfConfig) -> dict:
    """Per-dataset provenance: source, raw-byte SHA-256, size.

    Hashes the raw source bytes (before column mapping/sampling) so the
    fingerprint identifies the input data, not derived rows. Reads go through
    the same capped loaders datasets use, so remote sources respect the
    security byte/timeout limits.
    """
    from . import datasets

    out = {}
    for name, ds_cfg in (config.datasets or {}).items():
        entry = {"source": ds_cfg.source, "sha256": None, "bytes": None}
        try:
            raw = datasets._read_bytes(ds_cfg.source)
            entry["sha256"] = _sha256(raw)
            entry["bytes"] = len(raw)
        except Exception:
            pass  # unreadable at report time — keep the source string
        out[name] = entry
    return out


def _target_entry(target) -> dict:
    """Summarize one target's provider/model/endpoint identity for provenance."""
    entry = {"name": target.name, "layer": target.layer}
    if isinstance(target, LlmApiTarget):
        entry.update(
            provider=target.provider,
            model=(target.params or {}).get("model"),
            host=_host(target.base_url),
            region=(target.aws.region if target.aws else None),
            api_version=(target.azure.api_version if target.azure else None),
        )
    elif isinstance(target, PipelineTarget):
        llm_refs = []
        for stage in target.pipeline or []:
            llm = getattr(stage, "llm", None) or {}
            if llm.get("provider") or llm.get("model"):
                llm_refs.append(
                    {
                        "stage": stage.stage,
                        "provider": llm.get("provider"),
                        "model": llm.get("model"),
                        "host": _host(llm.get("base_url")),
                    }
                )
        entry.update(dataset=target.dataset, llm_refs=llm_refs or None)
    elif isinstance(target, AppEndpointTarget):
        entry.update(
            transport=target.transport,
            method=target.method,
            host=_host(target.base_url),
            endpoint=target.endpoint,
        )
    elif isinstance(target, AppObjectTarget):
        entry.update(framework=target.framework, python_path=target.python_path, call=target.call)
    return entry


def _load_generator() -> dict:
    """Environment facts about the machine generating the load."""
    info = {
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node() or None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["mem_total_gb"] = round(vm.total / (1024**3), 2)
    except Exception:
        info["mem_total_gb"] = None
    return info


def build(config: PerfConfig, config_path: str | None = None) -> dict:
    """Assemble the reproducibility manifest for one benchmark run."""
    pricing = getattr(config.report, "pricing", None)
    return {
        "run_id": str(uuid.uuid4()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "framework": {
            "name": "llm-perf",
            "version": _dist_version("llm-perf"),
            "locust": _dist_version("locust"),
            "python": sys.version.split()[0],
        },
        "git": _git_info(),
        "config": {
            "path": config_path,
            "sha256": _config_hash(config),
        },
        "datasets": _dataset_entries(config),
        "targets": [_target_entry(t) for t in (config.targets or [])],
        "load": {
            "profile": config.load.profile,
            "users": config.load.users,
            "spawn_rate": config.load.spawn_rate,
            "duration": config.load.duration,
            "run_time": config.load.run_time,
        },
        "load_generator": _load_generator(),
        "pricing": (
            {
                **pricing.model_dump(mode="json"),
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if pricing
            else None
        ),
    }
