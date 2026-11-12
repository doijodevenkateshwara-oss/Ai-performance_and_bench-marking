"""Capture a screenshot of the Locust web UI during a live run vs the mock server."""

import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_e2e import MockOpenAI  # noqa: E402

PORT = 8399
OUT = ROOT / "docs" / "screenshots" / "locust-ui.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

srv = HTTPServer(("127.0.0.1", 0), MockOpenAI)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/v1"

cfg_path = ROOT / "docs" / "screenshots" / "_shot_config.yaml"
cfg_path.write_text(f"""
observability:
  otel:
    enabled: false
targets:
  - name: mock_openai
    layer: llm_api
    provider: openai
    base_url: {base}
    stream: false
    params: {{model: mock-model, max_tokens: 32}}
    prompts: ["Say hello", "What is 2+2?", "Explain load testing"]
scenarios:
  - name: chat_load
    target: mock_openai
    weight: 100
load:
  users: 5
  spawn_rate: 5
""")

env = dict(os.environ, LLM_PERF_CONFIG=str(cfg_path))
proc = subprocess.Popen(  # noqa: S603 - launches our own locust CLI
    [str(ROOT / ".venv/bin/locust"), "-f", "locustfile.py", "--web-port", str(PORT)],
    cwd=ROOT,
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("locust UI never came up")

    data = urllib.parse.urlencode(
        {
            "user_count": 8,
            "spawn_rate": 8,
            "host": "http://localhost",
        }
    ).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/swarm", data=data)
    print("swarm:", urllib.request.urlopen(req, timeout=10).status)

    time.sleep(12)  # let stats accumulate

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print("saved", OUT)
finally:
    proc.terminate()
    proc.wait(timeout=10)
    srv.shutdown()
