# llm-perf

Multi-layer LLM/agent performance testing framework built on **Locust** + **OpenTelemetry**.

Test LLM systems at multiple layers from one YAML config:

| Layer | What it tests | Metrics |
|---|---|---|
| `llm_api` | Direct vendor calls — OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, generic HTTP, gRPC | latency, TTFT, tokens/sec, token usage, cost |
| `pipeline` | Declarative RAG/agent pipelines (retrieve → rerank → generate, agent loops) | e2e + per-stage latency, context recall |
| `app_endpoint` | Your deployed app over HTTP/gRPC (black-box) | latency, error rate, OTel trace correlation |
| `app_object` | Your LangChain / LlamaIndex / Python object in-process | latency, output quality |

Plus: dataset-driven prompts, hybrid quality scoring (heuristics + LLM-as-judge),
SLO thresholds with pass/fail gating and webhook alerts, OTel traces + metrics,
end-of-run CSV/JSON/HTML reports, and a Grafana dashboard.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional extras:
# pip install -e ".[langchain,llamaindex,s3,gcs]"
```

## Quick start

```bash
cp .env.example .env   # fill in API keys

# validate a config
llm-perf validate -c configs/openai_example.yaml

# run headless for 2 minutes with 10 users
llm-perf run -c configs/openai_example.yaml --headless -u 10 -t 2m

# or use the standard locust CLI directly (web UI on :8089)
LLM_PERF_CONFIG=configs/openai_example.yaml locust -f llm_perf.locustfile
```

Reports land in `report.output_dir` (CSV/JSON/HTML). Threshold failures exit
non-zero for CI (`thresholds.on_fail: fail`).

## Observability stack (optional)

```bash
docker compose up -d     # otel-collector + prometheus + jaeger + grafana
# Grafana: http://localhost:3000  (dashboard "llm-perf")
# Jaeger:  http://localhost:16686
```

Point `observability.otel.endpoint` at any OTLP backend (default
`http://localhost:4317`). GenAI semantic-convention attributes are emitted
when `genai_semconv: true`.

## Config tour

See `configs/` — one example per provider plus `rag_example.yaml` and
`grpc_example.yaml`. Key sections:

- `targets[]` — each has `layer`, provider/transport details, `dataset` or
  `prompts`, `params`, `stream`, `think_time`, `vars` for `{var}` templating.
- `scenarios[]` — bind targets with weights, per-scenario `params` overrides,
  and `quality` rules (heuristics + judge + `sample_rate`).
- `datasets{}` — named prompt stores (yaml/csv/jsonl, local/URL/S3/GCS) with
  sampling and `{var}` pools. Fields `expected`/`expected_context` enable
  answer-checking and RAG context-recall.
- `load` — users/spawn_rate/duration or staged ramps.
- `thresholds` — per-scenario + global SLO rules (`p95_ms`, `ttft_p95_ms`,
  `max_error_rate`, `min_quality_score`, per-`stage` rules), `on_fail`,
  `alerts` (slack/pagerduty/generic webhook, URLs via env vars).
- `report` — formats, output dir, percentiles, optional pricing for cost est.

## Datasets

```bash
llm-perf dataset list -c configs/openai_example.yaml
llm-perf dataset show qa_pairs -c configs/openai_example.yaml
llm-perf dataset sample qa_pairs -c configs/openai_example.yaml -n 5
llm-perf dataset import --from traces.jsonl --to datasets/imported.jsonl
llm-perf dataset validate datasets/qa_pairs.jsonl
```

Import production traces: `--trace-format langsmith|langfuse` or generic JSONL
with `--map prompt=inputs.question,expected=outputs.answer`.

## Quality scoring

Per scenario, `quality:` combines:

- **heuristics** (free): non-empty, min/max chars, required/forbidden regex,
  JSON validity, contains, exact match vs dataset `expected`, refusal detection.
- **judge** (sampled via `sample_rate`): sends prompt+response+rubric to any
  provider; JSON score parsing with normalization to 0..1.
- **rag**: `context_recall` vs dataset `expected_context` when the pipeline
  returns retrieved texts.

## Thresholds & alerts

```yaml
thresholds:
  global: { max_error_rate: 0.01, min_throughput_rps: 5 }
  scenarios:
    chat_load:
      p95_ms: 8000
      ttft_p95_ms: 1500
      min_quality_score: 0.7
      stages: { retrieve: { p95_ms: 300 } }
  on_fail: fail           # or warn
  alerts:
    - type: slack
      url_env: SLACK_WEBHOOK_URL
      on: [threshold_breach, test_complete]
```

Threshold state is also exported as `llm_perf_threshold_pass` gauges for
Grafana/Alertmanager mid-run alerting.

## CLI

```
llm-perf run -c cfg.yaml [--users N] [--spawn-rate R] [--duration 5m] [--headless] [--no-alerts]
llm-perf validate -c cfg.yaml [--skip-connectivity]
llm-perf proto-gen -p protos/my_service.proto
llm-perf quality-check -c cfg.yaml --response-file out.txt
llm-perf dataset list|show|sample|import|validate
```

## Project layout

```
llm_perf/
  config.py        # pydantic config schema
  cli.py           # typer CLI
  locustfile.py    # locust entry (generated users + lifecycle hooks)
  scenarios.py     # weighted User/TaskSet generation
  tasks.py         # per-request pipeline + quality hook
  metrics.py       # OTel instruments + Locust stats bridge
  otel/            # tracer/meter setup, GenAI semconv
  providers/       # openai, azure, anthropic, bedrock, http, grpc adapters
  transports/      # http + grpc for app_endpoint layer
  layers/          # llm_api, pipeline, app_endpoint, app_object
  pipelines/       # declarative stage engine (retrieve/rerank/generate/agent)
  adapters/        # langchain/llamaindex helpers
  quality/         # heuristics, judge, rag recall, scorer
  datasets.py      # dataset loaders/samplers/trace import
  prompts.py       # prompt pools + {var} templating
  thresholds.py    # SLO evaluation + webhooks
  reporters.py     # CSV/JSON/HTML reports
  templates/       # HTML report template
```

## Notes & limitations (v0.1)

- `agent` pipeline stage is a Phase-2 stub (raises a clear error).
- gRPC targets need a compiled proto (`grpc.proto_path`) or use the bundled
  minimal Triton `protos/inference.proto`; field mapping is best-effort.
- Bedrock uses the sync boto3 client in a threadpool.
- Streaming token usage requires provider support (`stream_options.include_usage`
  for OpenAI; Anthropic `message_delta`; Bedrock `metadata` events).
- Secrets come from env vars / cloud credential chains only — never config files.
