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


class EventLoopLagProbe:
    """Measures scheduler/event-loop lag — how late sleep() wakes up.

    Under Locust (gevent) or asyncio, blocking work delays greenlet
    resumption; TTFT/ITL chunk timestamps are taken when the greenlet runs,
    so high lag inflates streaming-timing metrics. This probe sleeps a fixed
    interval and records overshoot — a direct measure of that inflation.
    """

    def __init__(self, interval_s: float = 0.05, window: int = 200):
        self.interval_s = interval_s
        self._window = window
        self._lags_ms: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._greenlet = None

    def start(self):
        if self._thread is not None or self._greenlet is not None:
            return
        try:
            import gevent

            # run as a greenlet on the main hub — measures the real event-loop lag
            self._greenlet = gevent.spawn(self._loop)
        except Exception:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._greenlet is not None:
            with contextlib.suppress(Exception):
                self._greenlet.join(timeout=2)
        if self._thread:
            self._thread.join(timeout=2)

    def _sleep(self, s: float):
        try:
            import gevent

            gevent.sleep(s)
        except Exception:
            self._stop.wait(s)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            self._sleep(self.interval_s)
            lag_ms = max((time.monotonic() - t0 - self.interval_s) * 1000.0, 0.0)
            self._lags_ms.append(lag_ms)
            if len(self._lags_ms) > self._window:
                del self._lags_ms[: len(self._lags_ms) - self._window]

    def summary(self) -> dict:
        if not self._lags_ms:
            return {}
        s = sorted(self._lags_ms)
        n = len(s)
        return {
            "samples": n,
            "interval_ms": self.interval_s * 1000.0,
            "avg_ms": round(sum(s) / n, 2),
            "p50_ms": round(s[int(n * 0.50)], 2),
            "p95_ms": round(s[min(int(n * 0.95), n - 1)], 2),
            "max_ms": round(s[-1], 2),
        }


def evaluate_resource_warnings(loadgen: dict, cfg) -> list[dict]:
    """Compare loadgen resource metrics against warn thresholds -> warning list."""
    if cfg is None or not getattr(cfg, "enabled", True) or not loadgen:
        return []
    out = []

    def _add(level, metric, value, threshold, message):
        out.append(
            {"level": level, "metric": metric, "value": value, "threshold": threshold, "message": message}
        )

    cpu_avg = loadgen.get("cpu_percent_avg")
    cpu_max = loadgen.get("cpu_percent_max")
    mem_max = loadgen.get("mem_percent_max")
    lag = (loadgen.get("scheduler_lag") or {}).get("p95_ms")

    if cpu_avg is not None and cpu_avg > cfg.cpu_avg_pct:
        _add(
            "warn",
            "cpu_percent_avg",
            cpu_avg,
            cfg.cpu_avg_pct,
            "load-generator CPU avg is high — latency/ITL may be inflated",
        )
    if cpu_max is not None and cpu_max > cfg.cpu_max_pct:
        _add(
            "warn",
            "cpu_percent_max",
            cpu_max,
            cfg.cpu_max_pct,
            "load-generator CPU peaked — check generator saturation",
        )
    if mem_max is not None and mem_max > cfg.mem_pct:
        _add(
            "warn",
            "mem_percent_max",
            mem_max,
            cfg.mem_pct,
            "load-generator memory pressure — risk of skewed results",
        )
    if lag is not None and lag > cfg.lag_p95_ms:
        _add(
            "warn",
            "scheduler_lag_p95_ms",
            lag,
            cfg.lag_p95_ms,
            "event-loop lag is high — TTFT/ITL chunk timestamps are scheduling-inclusive "
            "and unreliable at this concurrency; reduce users or shard with --master/--workers",
        )
    return out


def scrape_prometheus(url: str, metrics: list[str], timeout_s: float = 5.0) -> dict:
    """Fetch a Prometheus text exposition and sum/avg the named metrics."""
    from .security import check_url

    out: dict[str, dict] = {}
    try:
        check_url(url)
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
