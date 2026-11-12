"""Integration tests for Prometheus metrics scraping (real local HTTP server).

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import pytest

from llm_perf.resources import scrape_prometheus


def test_scrape_prometheus(tmp_path):
    """scrape_prometheus sums named metrics from exposition text."""
    body = """
# HELP http_requests_total Total requests
http_requests_total{code="200"} 100
http_requests_total{code="500"} 5
process_cpu_seconds_total 12.5
"""
    p = tmp_path / "metrics.txt"
    p.write_text(body)

    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        """Stub HTTP handler serving a Prometheus exposition."""

        def do_GET(self):
            """Serve canned metrics text."""
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            """Silence handler logging."""

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = scrape_prometheus(f"http://127.0.0.1:{srv.server_port}/metrics", ["http_requests_total"])
        assert out["http_requests_total"]["sum"] == pytest.approx(105.0)
        assert out["http_requests_total"]["samples"] == 2
        # unlisted metric excluded
        assert "process_cpu_seconds_total" not in out
    finally:
        srv.shutdown()


def test_scrape_prometheus_error():
    """Unreachable scrape URLs return an error dict."""
    out = scrape_prometheus("http://127.0.0.1:1/metrics", [])
    assert "error" in out
