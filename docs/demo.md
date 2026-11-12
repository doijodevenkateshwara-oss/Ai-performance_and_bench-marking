# Live Demo: Observability Stack & Dashboards

This guide starts the local observability stack and runs a mock load test so
every dashboard shows real data. No cloud credentials or API keys are needed —
the test runs against a local mock OpenAI server.

## 1. Start the observability stack

The stack is defined in `docker-compose.yml` and consists of four services:

| Service         | Image                                    | Purpose                                        |
| --------------- | ---------------------------------------- | ---------------------------------------------- |
| `otel-collector`| `otel/opentelemetry-collector-contrib`   | Receives OTLP traces/metrics from the test run |
| `prometheus`    | `prom/prometheus`                        | Scrapes collector metrics every 15s            |
| `jaeger`        | `jaegertracing/all-in-one`               | Stores and displays distributed traces         |
| `grafana`       | `grafana/grafana`                        | Pre-provisioned LLM performance dashboard       |

Start everything:

```bash
docker compose up -d
```

Verify all four containers are healthy:

```bash
docker compose ps
```

Expected output (all `Up`):

```
performance-testing-llm-ai-grafana-1          Up   0.0.0.0:3000->3000/tcp
performance-testing-llm-ai-jaeger-1           Up   0.0.0.0:16686->16686/tcp
performance-testing-llm-ai-otel-collector-1   Up   0.0.0.0:4317-4318->4317-4318/tcp, 0.0.0.0:8889->8889/tcp
performance-testing-llm-ai-prometheus-1       Up   0.0.0.0:9090->9090/tcp
```

If `otel-collector` exits immediately, check its logs — the config lives in
`otel-collector-config.yaml` (mounted into the container).

```bash
docker compose logs otel-collector
```

## 2. Dashboard URLs

Once the stack is running, open:

| Dashboard      | URL                              | What to look at                                                        |
| -------------- | -------------------------------- | ---------------------------------------------------------------------- |
| **Grafana**    | http://localhost:3000            | Dashboards → *LLM Performance Testing* — request rate, latency percentiles, TTFT, tokens/sec, error rate |
| **Prometheus** | http://localhost:9090            | Status → Targets (should show `otel-collector` UP); Graph for ad-hoc PromQL |
| **Jaeger**     | http://localhost:16686           | Select service `llm-perf-demo` (or your `service_name`) → Find Traces  |
| **OTel metrics** | http://localhost:8889/metrics | Raw Prometheus exposition endpoint scraped by Prometheus               |
| **Locust UI**  | http://localhost:8089            | Live load-test stats — only while a Locust run is active               |

Grafana needs no login setup: the compose file provisions the datasource and
dashboard automatically (`grafana/provisioning/`). The default admin login is
`admin` / `admin` if prompted.

## 3. Run a demo test that populates the dashboards

The framework exports OpenTelemetry traces and metrics to the collector at
`http://localhost:4317` whenever `observability.otel.enabled: true` in the
config (this is the default in the example configs).

### Option A — mock server (no API keys, no cost)

For a no-cost demo, run against a local mock OpenAI-compatible server. Any
config with `observability.otel.enabled: true` and `stream: true` works —
example (`configs/demo_otel.yaml`):

```yaml
observability:
  otel:
    enabled: true
    endpoint: http://localhost:4317
    service_name: llm-perf-demo
    sample_ratio: 1.0
    genai_semconv: true

targets:
  - name: mock_openai
    layer: llm_api
    provider: openai
    base_url: http://127.0.0.1:PORT/v1   # your mock/real endpoint
    stream: true
    params:
      model: mock-model
      max_tokens: 64
    prompts:
      - "Say hello"
      - "What is 2+2?"
    timeout_s: 30

scenarios:
  - name: stream_smoke
    target: mock_openai
    weight: 1

load:
  users: 3
  spawn_rate: 3
  duration: 60s
  ramp_up: 5s
  ramp_down: 5s

report:
  formats: [json]
  output_dir: results/demo
```

Then run headless:

```bash
python -m locust -f locustfile.py --headless --run-time 75s
```

Or with the web UI (watch http://localhost:8089 while Grafana/Jaeger fill in):

```bash
python -m locust -f locustfile.py
```

### Option B — real provider

Point `base_url` at a real endpoint and set the API key env var, e.g. copy
`configs/openai_example.yaml`, keep `observability.otel` enabled, and run:

```bash
export OPENAI_API_KEY=sk-...
llm-perf run -c configs/openai_example.yaml
```

## 4. What you should see

- **Prometheus**: `llm_perf_request_total`, `llm_perf_request_duration_seconds`,
  `llm_perf_ttft_seconds`, `llm_perf_tpot_seconds`,
  `llm_perf_tokens_per_second`, `llm_perf_users`, etc. Metrics export every
  15s; Prometheus scrapes every 15s — allow ~30s after the run starts.
- **Grafana**: panels populate once Prometheus has scraped a few intervals.
- **Jaeger**: one trace per sampled request under service `llm-perf-demo`,
  with GenAI semantic-convention attributes (model, token usage, TTFT).

## 5. Tear down

```bash
docker compose down          # stop and remove containers (keeps volumes)
docker compose down -v       # also wipe Prometheus/Grafana data
```
