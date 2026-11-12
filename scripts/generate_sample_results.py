"""Generate published sample results against a local mock OpenAI server.

Runs a real headless Locust test (streaming SSE with artificial chunk delays)
and writes the CSV/JSON/HTML reports into docs/sample-results/.

Usage: .venv/bin/python scripts/generate_sample_results.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docs" / "sample-results"
CHUNK_DELAY_S = 0.03  # 30ms between SSE chunks -> visible ITL/TPOT
N_CHUNKS = 12


class MockOpenAIStream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        model = req.get("model", "mock-model")
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        stream = bool(req.get("stream"))
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            chunks = [
                "Hello",
                " there",
                ",",
                " this",
                " is",
                " a",
                " mock",
                " streaming",
                " response",
                " from",
                " llm",
                "-perf",
                ".",
            ]
            try:
                for c in chunks[:N_CHUNKS]:
                    payload = {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(CHUNK_DELAY_S)
                usage = {"prompt_tokens": 12, "completion_tokens": N_CHUNKS, "total_tokens": 12 + N_CHUNKS}
                final = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage,
                }
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            resp = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "mock response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            }
            payload = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, *a):
        pass


def main() -> int:
    from http.server import ThreadingHTTPServer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIStream)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "observability": {"otel": {"enabled": False}},
        "targets": [
            {
                "name": "mock_openai",
                "provider": "openai",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "stream": True,
                "params": {"model": "mock-model", "max_tokens": 64},
                "prompts": ["Say hello", "What is 2+2?", "Explain TTFT"],
                "timeout_s": 30,
            }
        ],
        "scenarios": [{"name": "stream_smoke", "target": "mock_openai", "weight": 1}],
        "load": {
            "users": 3,
            "spawn_rate": 3,
            "profile": "smoke",
            "duration": "20s",
            "ramp_up": "5s",
            "ramp_down": "5s",
        },
        "report": {
            "formats": ["csv", "json", "html"],
            "output_dir": str(OUT_DIR),
            "collect_resources": True,
            "history": False,
            "baseline": "none",
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        import yaml

        yaml.safe_dump(cfg, f)
        cfg_path = f.name

    env = dict(os.environ)
    env["LLM_PERF_CONFIG"] = cfg_path
    locustfile = REPO / "locustfile.py"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(locustfile),
            "--headless",
            "--run-time",
            "40s",
        ],
        check=False,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
    srv.shutdown()
    os.unlink(cfg_path)

    produced = sorted(p.name for p in OUT_DIR.glob("llm-perf-*"))
    print("SAMPLE_RESULTS", produced)
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
