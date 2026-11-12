"""Subprocess runner for the e2e Locust test. Usage: e2e_runner.py <base_url>"""

import json
import sys
import time

base_url = sys.argv[1]

from locust.env import Environment  # noqa: E402  (import order matters for gevent)

from llm_perf.config import PerfConfig  # noqa: E402
from llm_perf.scenarios import build_users  # noqa: E402

cfg = PerfConfig.model_validate(
    {
        "observability": {"otel": {"enabled": False}},
        "targets": [
            {
                "name": "mock_openai",
                "provider": "openai",
                "base_url": base_url,
                "stream": False,
                "params": {"model": "mock-model", "max_tokens": 32},
                "prompts": ["Say hello", "What is 2+2?"],
            }
        ],
        "scenarios": [{"name": "smoke", "target": "mock_openai", "weight": 1}],
        "load": {"users": 2, "spawn_rate": 2, "duration": "4s"},
    }
)

users = build_users(cfg)
env = Environment(user_classes=users)
runner = env.create_local_runner()
runner.start(2, spawn_rate=2)
time.sleep(5)
runner.stop()

stats = env.stats
entries = list(stats.entries.values())
methods = {e.method for e in entries}
result = {
    "total_requests": stats.total.num_requests,
    "total_failures": stats.total.num_failures,
    "has_tokens_stat": "TOKENS" in methods,
    "has_ttft_or_tokens": bool({"TTFT", "TOKENS", "PROMPT_TOKENS"} & methods),
    "scenarios": sorted({e.name for e in entries}),
}
runner.quit()
print("E2E_RESULT " + json.dumps(result))
