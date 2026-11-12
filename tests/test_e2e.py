"""End-to-end smoke: run a real Locust load test in a subprocess against a mock server.

Locust monkeypatches the process with gevent, which conflicts with pytest's
already-imported modules — and real usage is a separate process anyway — so the
runner executes `tests/e2e_runner.py` via subprocess.


Author: Venkateshwara Doijode
"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


class MockOpenAI(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible mock server for e2e runs."""

    def do_POST(self):
        """Serve a canned chat.completion response."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        if self.path.endswith("/chat/completions"):
            resp = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": req.get("model", "mock"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "mock response to: "
                            + str(req.get("messages", [{}])[-1].get("content", ""))[:50],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            }
            payload = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        """Silence BaseHTTPRequestHandler logging."""


@pytest.fixture()
def mock_server():
    """Fixture: start MockOpenAI on an ephemeral port."""
    srv = HTTPServer(("127.0.0.1", 0), MockOpenAI)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_locust_run_against_mock(mock_server):
    """Full locust run against the mock produces a report."""
    runner = Path(__file__).parent / "e2e_runner.py"
    proc = subprocess.run(  # noqa: S603 - runs our own runner script
        [sys.executable, str(runner), mock_server],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        cwd=str(Path(__file__).parent.parent),
    )
    assert proc.returncode == 0, f"runner failed:\n{proc.stdout}\n{proc.stderr}"
    result_line = next((line for line in proc.stdout.splitlines() if line.startswith("E2E_RESULT ")), None)
    assert result_line, f"no result line in stdout:\n{proc.stdout[-2000:]}"
    out = json.loads(result_line[len("E2E_RESULT ") :])
    assert out["total_requests"] > 0
    assert out["has_tokens_stat"] is True
    assert out["has_ttft_or_tokens"] is True


def test_layer_direct_call(mock_server):
    """Layer-level smoke: call LlmApiLayer directly without Locust."""
    from llm_perf.config import LlmApiTarget
    from llm_perf.layers.llm_api import LlmApiLayer
    from llm_perf.prompts import PromptPicker

    target = LlmApiTarget(
        name="t", provider="openai", base_url=mock_server, params={"model": "m"}, prompts=["hi"]
    )
    layer = LlmApiLayer(target, PromptPicker(prompts=["hi"]))
    res = layer.run("hello", None, {}, scenario_name="direct")
    assert res.error is None
    assert "mock response" in res.output_text
    assert res.response.completion_tokens == 8
