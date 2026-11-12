"""Capture screenshots of all observability dashboard pages into docs/screenshots/.

Requires the docker compose stack to be running (docker compose up -d) with at
least one completed test run so panels/traces contain data.

Usage: .venv/bin/python scripts/screenshot_dashboards.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

VIEWPORT = {"width": 1600, "height": 1000}


def shot(page, url: str, path: Path, wait_ms: int = 4000, full_page: bool = False) -> None:
    print(f"capturing {url} -> {path.name}")
    page.goto(url, wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(wait_ms)
    page.screenshot(path=str(path), full_page=full_page)


def grafana(page) -> None:
    # Grafana may show a login page; default admin/admin is provisioned by compose.
    page.goto("http://localhost:3000/login", wait_until="networkidle", timeout=30000)
    if page.locator('input[name="user"]').count():
        page.fill('input[name="user"]', "admin")
        page.fill('input[name="password"]', "admin")
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")
    # Grafana scrolls inside an inner container, so full_page doesn't help —
    # use a tall viewport instead so all 10 panels fit in one shot.
    page.set_viewport_size({"width": 1600, "height": 3400})
    shot(
        page,
        "http://localhost:3000/d/llm-perf-main/llm-perf-e28094-llm-performance-and-quality",
        OUT / "grafana-llm-perf.png",
        wait_ms=8000,
    )
    page.set_viewport_size(VIEWPORT)


def prometheus(page) -> None:
    # Graph page pre-loaded with an llm_perf query via URL params.
    expr = "histogram_quantile(0.95, sum by (le) (rate(llm_perf_request_duration_seconds_bucket[5m])))"
    url = (
        "http://localhost:9090/graph"
        "?g0.expr=" + urllib.parse.quote(expr) + "&g0.tab=0"
    )
    shot(page, url, OUT / "prometheus-graph.png", wait_ms=6000)
    # Targets/status page showing the otel-collector scrape target.
    shot(page, "http://localhost:9090/targets", OUT / "prometheus-targets.png", wait_ms=4000)


def jaeger(page) -> None:
    # Search page filtered to the demo service.
    shot(
        page,
        "http://localhost:16686/search?service=llm-perf-demo&lookback=1h",
        OUT / "jaeger-search.png",
        wait_ms=6000,
    )
    # Trace detail page for the most recent trace, if any exist.
    try:
        data = json.loads(
            urllib.request.urlopen(
                "http://localhost:16686/api/traces?service=llm-perf-demo&limit=1", timeout=10
            ).read()
        )
        traces = data.get("data") or []
        if traces:
            trace_id = traces[0]["traceID"]
            shot(page, f"http://localhost:16686/trace/{trace_id}", OUT / "jaeger-trace.png", wait_ms=5000)
    except Exception as exc:
        print(f"jaeger trace detail skipped: {exc}", file=sys.stderr)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()
        for fn in (grafana, prometheus, jaeger):
            try:
                fn(page)
            except Exception as exc:  # keep going; one dashboard down shouldn't block others
                print(f"FAILED {fn.__name__}: {exc}", file=sys.stderr)
        browser.close()
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
