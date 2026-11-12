"""Resource utilization monitoring.

- Load-generator host metrics via psutil (CPU%, memory, load avg) sampled on a
  greenlet/thread during the run.
- Optional target-side scrape: pull selected metrics from a Prometheus
  /metrics endpoint configured under report.scrape.
"""

from __future__ import annotations

import contextlib
import threading
import time
from urllib.request import urlopen

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a hard dep but stay defensive
    psutil = None


class HostSampler:
    """Samples local CPU/memory every `interval_s` until stopped."""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process() if psutil else None
        if self._proc is not None:
            self._proc.cpu_percent(None)  # prime

    def start(self):
        if psutil is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self._sample()
            self._stop.wait(self.interval_s)

    def _sample(self):
        vm = psutil.virtual_memory()
        self._samples.append(
            {
                "ts": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_percent": vm.percent,
                "mem_used_mb": vm.used / 1e6,
                "load_avg_1m": (psutil.getloadavg() or [None])[0] if hasattr(psutil, "getloadavg") else None,
            }
        )

    def summary(self) -> dict:
        if not self._samples:
            return {}
        cpus = [s["cpu_percent"] for s in self._samples]
        mems = [s["mem_percent"] for s in self._samples]
        return {
            "samples": len(self._samples),
            "interval_s": self.interval_s,
            "cpu_percent_avg": round(sum(cpus) / len(cpus), 1),
            "cpu_percent_max": round(max(cpus), 1),
            "mem_percent_avg": round(sum(mems) / len(mems), 1),
            "mem_percent_max": round(max(mems), 1),
            "mem_used_mb_max": round(max(s["mem_used_mb"] for s in self._samples), 1),
        }


def scrape_prometheus(url: str, metrics: list[str], timeout_s: float = 5.0) -> dict:
    """Fetch a Prometheus text exposition and sum/avg the named metrics."""
    out: dict[str, dict] = {}
    try:
        with urlopen(url, timeout=timeout_s) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        return {"error": str(e), "url": url}
    wanted = set(metrics or [])
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, val = parts
        # strip labels: metric{...} -> metric
        base = name.split("{", 1)[0]
        if wanted and base not in wanted:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        sums[base] = sums.get(base, 0.0) + v
        counts[base] = counts.get(base, 0) + 1
    for base, total in sums.items():
        out[base] = {"sum": total, "samples": counts[base], "avg": total / counts[base]}
    return out


def collect_target_metrics(scrape_configs: list) -> dict:
    """scrape_configs: list of {name, url, metrics, timeout_s}."""
    results = {}
    for cfg in scrape_configs or []:
        name = getattr(cfg, "name", None) or (cfg.get("name") if isinstance(cfg, dict) else "target")
        url = getattr(cfg, "url", None) or (cfg.get("url") if isinstance(cfg, dict) else None)
        metrics = getattr(cfg, "metrics", None) or (cfg.get("metrics") if isinstance(cfg, dict) else None)
        timeout = getattr(cfg, "timeout_s", 5.0) or 5.0
        if not url:
            continue
        results[name] = scrape_prometheus(url, metrics or [], timeout_s=timeout)
    return results
