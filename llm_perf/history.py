"""Run history store — every report JSON is kept under results/history/.

Enables previous-run deltas, designated baselines, and regression detection.


Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import PerfConfig


def history_dir(config: PerfConfig) -> Path:
    """Directory holding per-run history JSON files."""
    return Path(config.report.output_dir) / "history"


def _index_path(config: PerfConfig) -> Path:
    """Path to the append-only history index (index.jsonl)."""
    return history_dir(config) / "index.jsonl"


def record_run(report: dict, config: PerfConfig) -> Path | None:
    """Persist a run's aggregate report + index entry. Returns the file path."""
    if not config.report.history:
        return None
    try:
        d = history_dir(config)
        d.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = d / f"run-{ts}.json"
        # avoid clobbering same-second runs
        i = 1
        while path.exists():
            path = d / f"run-{ts}-{i}.json"
            i += 1
        path.write_text(json.dumps(report, indent=2, default=str))
        totals = report.get("totals") or {}
        entry = {
            "ts": report.get("generated_at"),
            "file": path.name,
            "requests": totals.get("requests"),
            "failures": totals.get("failures"),
            "error_rate": totals.get("error_rate"),
            "profile": (report.get("load") or {}).get("profile"),
        }
        with _index_path(config).open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return path
    except Exception:
        return None


def load_history(config: PerfConfig, limit: int = 100) -> list[dict]:
    """Return history entries newest-first (full report dicts)."""
    d = history_dir(config)
    if not d.exists():
        return []
    files = sorted(d.glob("run-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in files:
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def previous_run(config: PerfConfig, before_ts: str | None = None) -> dict | None:
    """Most recent recorded run (optionally strictly before `before_ts`)."""
    runs = load_history(config)
    if before_ts:
        runs = [r for r in runs if (r.get("generated_at") or "") < before_ts]
    return runs[0] if runs else None


def baseline_path(config: PerfConfig) -> Path:
    """Path to the designated-baseline pointer file."""
    return Path(config.report.output_dir) / "baseline.json"


def mark_baseline(config: PerfConfig) -> Path | None:
    """Designate the latest history run as the comparison baseline."""
    latest = load_history(config, limit=1)
    if not latest:
        return None
    path = baseline_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"baseline": latest[0]}, indent=2, default=str))
    return path


def resolve_baseline(config: PerfConfig, current: dict) -> dict | None:
    """Resolve report.baseline -> a baseline report dict (or None)."""
    b = config.report.baseline
    if not b or b == "none":
        return None
    try:
        if b == "designated":
            p = baseline_path(config)
            if p.exists():
                return json.loads(p.read_text()).get("baseline")
            return None
        if b == "previous":
            return previous_run(config, before_ts=current.get("generated_at"))
        # explicit path to a run JSON or a baseline.json pointer
        p = Path(b)
        if p.exists():
            data = json.loads(p.read_text())
            return data.get("baseline") or data
        return None
    except Exception:
        return None
