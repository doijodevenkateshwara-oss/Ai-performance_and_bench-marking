# Demo evidence pack

Dummy targets + a local mock server that exercise every supported layer and
metric family **without API keys or external services**. Run artifacts land in
`results/demo-evidence/` (gitignored); committed copies live under
`docs/sample-results/`.

## Contents

| File | Purpose |
|---|---|
| `mock_server.py` | OpenAI-compatible mock: unary JSON, SSE streaming, `tool_calls`, `/search` for agent HTTP tools, `/health` |
| `dummy_tools.py` | Python function tools for the agent stage (`echo`, `word_count`) |
| `dummy_app.py` | In-process app object for the `app_object` layer (`DummyApp.invoke`) |
| `all_layers.yaml` | One config hitting all four layers + streaming + quality + stages |
| `thresholds_security.yaml` | Three-state thresholds (pass/fail/not_evaluated) + SSRF allowlist demo |

## Run it

```bash
# 1. start the mock (terminal 1)
python demo/mock_server.py            # listens on http://127.0.0.1:8901/v1

# 2. materialize effective config, then run locust (terminal 2)
python - <<'EOF'
import yaml
from llm_perf.config import PerfConfig
for name in ("all_layers", "thresholds_security"):
    cfg = PerfConfig.from_yaml(f"demo/{name}.yaml")
    open(f"demo/.llm_perf_effective_{name}.yaml", "w").write(
        yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True), sort_keys=False))
EOF

LLM_PERF_CONFIG=demo/.llm_perf_effective_all_layers.yaml \
  locust -f locustfile.py --headless -u 6 -r 6 -t 45s

LLM_PERF_CONFIG=demo/.llm_perf_effective_thresholds_security.yaml \
  locust -f locustfile.py --headless -u 4 -r 4 -t 30s   # exits 1: FAIL rule
```

> `llm-perf run -c demo/all_layers.yaml --headless` hits a known
> gevent/urllib3 monkey-patch ordering issue when llm_perf modules are
> imported before locust; running `locust` directly avoids it.

## What the runs demonstrate

**`all_layers.yaml`** — 6 scenarios, ~1.1k requests, exit 0:

- `llm_api` unary + streaming (TTFT/TPOT/ITL/GEN_DURATION/tokens-per-sec)
- `pipeline` RAG: retrieve → generate with per-stage p95 metrics + CONTEXT_RECALL
- `pipeline` agent: tool-call loop (http `/search` + python `echo`), AGENT_STEPS/TOOL_CALLS/SUCCESS stats
- `app_endpoint` HTTP with Jinja2 request template + response mapping
- `app_object` in-process generic-framework call
- Quality: heuristics + LLM-judge (mock returns a parseable score) + RAG recall
- OTel spans exported to the local collector when `docker compose up -d` is running

**`thresholds_security.yaml`** — exit 1, all three rule states in one report:

- `pass` — `ok_load.p95_ms`, `ok_load.max_error_rate`
- `fail` — `blocked_load.max_error_rate` (1.0 > 0.5; every request to
  `169.254.169.254` blocked by `security.url_allowlist`)
- `not_evaluated` — `ok_load.ttft_p95_ms` (unary target, no TTFT samples)

## Coverage map (what is and isn't evidenced here)

| Category | Evidence in these runs |
|---|---|
| Provider contracts | openai unary + streaming + tool_calls via mock |
| Streaming | SSE chunks, TTFT, TPOT, ITL, token usage |
| Metrics | latency percentiles, token accounting, stage metrics, agent stats |
| Workloads | smoke profile, ramp_up/steady/ramp_down phases, per-scenario weights |
| Pipelines | RAG retrieve→generate, agent tool loop |
| Agents | http + function tools, iteration/loop caps |
| Security | URL allowlist blocking (SSRF), untrusted-config audit warnings |
| Regression/thresholds | pass/fail/not_evaluated states, nonzero CI exit |
| Distributed | master/worker flags exist; not exercised here (single process) |
| Deployment | compose stack documented; no k8s manifests |
| Long-duration/soak | not exercised (45s runs); `profile: soak` exists in config |
