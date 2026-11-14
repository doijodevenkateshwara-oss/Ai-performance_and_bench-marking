# How to run llm-perf

## Prerequisites

- Python 3.10+
- `pip` and `venv`
- API keys for the providers you want to test (stored in env vars only)
- Docker (optional, for the local observability stack)

## Install

```bash
cd performance-testing-llm-ai
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[langchain]"      # app_object layer for LangChain
pip install -e ".[llamaindex]"     # app_object layer for LlamaIndex
pip install -e ".[s3,gcs]"         # dataset sources on S3/GCS
```

Configure secrets:

```bash
cp .env.example .env
# fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
# then either export them or use a tool like direnv
```

## Validate a config

```bash
llm-perf validate -c configs/openai_example.yaml
# skip connectivity/env checks for offline validation
llm-perf validate -c configs/openai_example.yaml --skip-connectivity
```

## Run a test

### Headless (CI-friendly)

```bash
llm-perf run -c configs/openai_example.yaml --headless -u 10 -t 2m
```

- `-u/--users` overrides `load.users`
- `--spawn-rate` overrides `load.spawn_rate`
- `-t/--duration` overrides `load.duration`
- `--headless` runs without the web UI

### With the Locust web UI

```bash
LLM_PERF_CONFIG=configs/openai_example.yaml locust -f locustfile.py
# open http://localhost:8089
```

Or via the CLI wrapper (uses `--web-port`, default 8089):

```bash
llm-perf run -c configs/openai_example.yaml --web-port 8089
```

### Workload profiles

Profiles expand into ramp-up → steady-state → ramp-down phases via a Locust
`LoadTestShape`:

```bash
# use the profile defined in the config
llm-perf run -c configs/profiles_example.yaml --headless

# override the profile on the CLI
llm-perf run -c configs/openai_example.yaml --profile stress --users 50 --headless
```

Available profiles: `smoke`, `baseline`, `load`, `stress`, `spike`, `soak`,
`scalability`, `capacity`, `custom` (see `configs/profiles_example.yaml` and
[README.md](../README.md#workload-profiles)).

### Run only some layers or scenarios

A single config can contain all four layers; filter at invocation time. For
example, with your own config that defines both `llm_api` and `pipeline`
targets (substitute the filename):

```bash
llm-perf run -c configs/my_stack.yaml --layers llm_api --headless
llm-perf run -c configs/my_stack.yaml --layers llm_api,pipeline --headless
llm-perf run -c configs/my_stack.yaml --scenarios rag_load --headless
```

### Resilience validation

Resilience uses the normal run command; `fault_injection` creates the failure
window and thresholds assert the observed retry/breaker/fallback behavior:

```bash
llm-perf validate -c configs/rag_resilience.yaml --skip-connectivity
llm-perf run -c configs/rag_resilience.yaml --headless
```

See [resilience.md](resilience.md) for supported retry classes, breaker states,
fallback types, fault windows, metrics, and black-box limitations.

### Availability SLO validation

Use `min_availability`, `min_functional_availability`, and
`min_non_degraded_availability` threshold rules to distinguish request success,
usable output, and primary-path completion. Reports include whole-run rates and
per-phase `slo_windows` when the workload has phases.

```bash
llm-perf run -c configs/rag_resilience.yaml --headless
# exits non-zero when thresholds.on_fail: fail and an SLO is breached
```

See [slo.md](slo.md) for metric definitions, configuration, report fields, and
SLA boundaries.

### Distributed run

```bash
# master
llm-perf run -c configs/openai_example.yaml --headless --master 0.0.0.0

# workers (repeat on each worker host)
llm-perf run -c configs/openai_example.yaml --headless --workers
```

## Common CLI commands

```bash
# benchmark bounded configuration candidates and write optimized.yaml
llm-perf optimize -c configs/my_pipeline.yaml --dry-run
llm-perf optimize -c configs/my_pipeline.yaml --trials 8

# inspect datasets
llm-perf dataset list -c configs/openai_example.yaml
llm-perf dataset show qa_pairs -c configs/openai_example.yaml
llm-perf dataset sample qa_pairs -c configs/openai_example.yaml -n 5

# import production traces into a dataset
llm-perf dataset import --from traces.jsonl --to datasets/imported.jsonl \
  --trace-format langsmith
# or generic JSONL with field mapping
llm-perf dataset import --from traces.jsonl --to datasets/imported.jsonl \
  --map prompt=inputs.question,expected=outputs.answer

# validate a standalone dataset file
llm-perf dataset validate datasets/qa_pairs.jsonl

# score a saved response offline against a config's quality rules
llm-perf quality-check -c configs/openai_example.yaml \
  --response-file response.txt --question "What is Kubernetes?"

# compile a .proto for gRPC targets
llm-perf proto-gen -p protos/inference.proto

# manage run history baselines
llm-perf baseline list -c configs/openai_example.yaml
llm-perf baseline mark -c configs/openai_example.yaml
```

## Observability stack (optional but recommended)

Start the local collector + Prometheus + Jaeger + Grafana (stack lives in
`deployments/`; the root `docker-compose.yml` is an `include:` shim):

```bash
docker compose up -d
# or explicitly: docker compose -f deployments/docker/docker-compose.yml up -d
```

Then run a test with `observability.otel.enabled: true` (the example configs
already do this). Open:

| UI | URL | Shows |
|---|---|---|
| Locust web UI | http://localhost:8089 | Live RPS, latency percentiles, failures, custom stats |
| Grafana | http://localhost:3000 | Provisioned `llm-perf` dashboard |
| Prometheus | http://localhost:9090 | Raw `llm_perf_*` metrics |
| Jaeger | http://localhost:16686 | Distributed traces per request/stage |

To run without telemetry, set `observability.otel.enabled: false` in the config.

## Dashboard screenshots

Committed screenshots of every dashboard live in `docs/screenshots/`:

- `grafana-llm-perf.png` — Grafana LLM Performance & Quality dashboard
- `prometheus-graph.png` / `prometheus-targets.png` — Prometheus graph and targets
- `jaeger-search.png` / `jaeger-trace.png` — Jaeger trace search and detail
- `locust-ui.png` — Locust live UI during a run

See [demo.md](demo.md) for the full walkthrough that produced them.

## Sample results without real credentials

`docs/sample-results/` contains committed CSV/JSON/HTML reports from a real
headless run against a local mock OpenAI streaming server using a `smoke`
profile — use them to see the full report shape (executive summary,
bottleneck analysis, SLO rollup, resource utilization, cost) without API
keys.

## Security

Configs can point the framework at arbitrary endpoints, import Python
modules, and fetch remote datasets — treat YAML configs as trusted code.
A prominent warning is printed at startup when a config uses powerful
features (Python imports, `request_template`, agent HTTP/function tools,
remote dataset URLs) unless you set `security.trusted: true`.

Guardrails live under `security:` in the config:

```yaml
security:
  allow_private_networks: true    # false = block loopback/RFC1918/link-local/reserved IPs (SSRF guard)
  url_allowlist: []               # e.g. ["api.openai.com", "*.corp.example.com"]; empty = no restriction
  python_execution: warn          # allow | warn | off — gates python_path imports & function tools
  dataset_max_bytes: 52428800     # 50 MiB cap per dataset source
  dataset_max_rows: 1000000
  dataset_timeout_s: 30
  redact: true                    # scrub credentials from error stats and span attributes
  trusted: false                  # true = silence untrusted-config warnings
```

Enforcement points:

- **All outbound HTTP** — provider endpoints, `app_endpoint` targets,
  agent `http` tools, pipeline `vector_store` http retrieves, dataset
  `http(s)` fetches, and `report.scrape` targets all pass through
  `check_url()`: scheme must be http/https, host must match
  `url_allowlist` (when set), and resolved IPs must be globally routable
  when `allow_private_networks: false`.
- **Python execution** — `app_object` `python_path` imports and agent
  `type: function` tools are gated by `python_execution`: `warn` (default)
  allows with a one-time warning, `allow` is silent, `off` blocks.
- **Datasets** — every source (local file, http(s), s3, gs) is capped by
  `dataset_max_bytes`/`dataset_max_rows`; remote fetches use
  `dataset_timeout_s` and the same URL policy.
- **Redaction** — `record_error`/`log_request_event`/span error attributes
  scrub `Bearer` tokens, `api_key=`/`token=`/`password=`-style params, URL
  userinfo, and common key shapes (`sk-`, `ghp_`, `AKIA`, private-key
  blocks) when `redact: true`.

Defaults preserve prior behavior: private networks reachable (the
framework exists to test internal services), Python tools allowed with a
warning, 50 MiB/1M-row dataset caps. Tighten for CI running configs from
untrusted sources. gRPC targets are not covered by the URL checks.

## CI usage

The repo's own CI (`.github/workflows/ci.yml`) runs:

```bash
pip install -e ".[dev]"
ruff check llm_perf tests
black --check llm_perf tests
for f in configs/*.yaml; do llm-perf validate -c "$f"; done
pytest -q
```

For your own CI, gate on thresholds/regressions:

```bash
llm-perf run -c configs/openai_example.yaml --headless -u 10 -t 5m
# exits non-zero if thresholds.on_fail: fail or regression.on_fail: fail
```

## Troubleshooting

- **Missing env vars**: `llm-perf validate` prints warnings for unset
  `api_key_env` variables and missing dataset sources.
- **Streaming metrics look inflated**: check `resources.loadgen` in the JSON
  report — high CPU or `scheduler_lag` means the load generator is the
  bottleneck, not the target. See [methodology.md](methodology.md).
- **gRPC errors**: run `llm-perf proto-gen -p <your.proto>` first and ensure
  `grpc.proto_path` points to the compiled proto.
- **Bedrock errors**: uses the AWS credential chain; set `AWS_REGION` and
  credentials in env or use `aws.auth: profile`.
