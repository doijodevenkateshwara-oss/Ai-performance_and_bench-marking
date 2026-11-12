# Architecture

`llm-perf` is a Locust-based load generator wrapped in a declarative YAML
configuration layer, with OpenTelemetry for live telemetry and a post-run
reporting/analysis pipeline.

## High-level diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YAML config (configs/*.yaml)                 │
│  observability · datasets · targets · scenarios · load · thresholds  │
│  report                                                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ PerfConfig.from_yaml()
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         llm-perf CLI (cli.py)                        │
│  validate / run / proto-gen / quality-check / dataset / baseline    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ writes .llm_perf_effective.yaml
                               │ sets LLM_PERF_CONFIG
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Locust runtime (locustfile.py)                    │
│  build_users() → weighted User classes per scenario                  │
│  LoadTestShape (optional) → ramp/hold/down phases                    │
│  events.test_start / test_stop / quitting                            │
└───────────────┬───────────────────────────────┬─────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────┐       ┌───────────────────────────────┐
│   Request execution path  │       │   Lifecycle hooks             │
│   scenarios.py → tasks.py │       │   otel setup, HostSampler,    │
│   → layers → providers/   │       │   EventLoopLagProbe, reports, │
│   pipelines/transports    │       │   thresholds, alerts, history │
└───────────┬───────────────┘       └───────────────┬───────────────┘
            │                                       │
            ▼                                       ▼
┌───────────────────────────┐       ┌───────────────────────────────┐
│  Metrics & traces         │       │  Reports & analysis           │
│  metrics.py + otel/       │       │  reporters.py, thresholds.py, │
│  Locust custom stats      │       │  analysis.py, history.py      │
└───────────┬───────────────┘       └───────────────┬───────────────┘
            │                                       │
            ▼                                       ▼
   OTLP/gRPC → collector → Prometheus/Grafana/Jaeger   CSV / JSON / HTML
```

## Config model

`llm_perf/config.py` defines the whole schema with Pydantic:

- `observability.otel` — enable/disable, endpoint, service name, sample ratio
- `datasets` — named dataset sources (local path, HTTP, S3, GCS)
- `targets` — the thing under test; each has a `layer`:
  - `llm_api` — direct vendor API (OpenAI, Anthropic, Azure, Bedrock, generic HTTP, gRPC)
  - `pipeline` — declarative multi-stage RAG/agent pipeline
  - `app_endpoint` — black-box HTTP/gRPC endpoint
  - `app_object` — in-process LangChain/LlamaIndex/Python object
- `scenarios` — weighted bindings from a target to load parameters, think time, quality rules
- `load` — users, spawn rate, duration, optional workload `profile`/`stages`
- `thresholds` — global/per-scenario SLO rules + alerts
- `report` — output formats, pricing, history/baseline/regression, resource collection/scraping

## Request lifecycle

1. `cli.run` loads the YAML, applies `--layers`/`--scenarios`/`--profile`/`--users` overrides,
   and writes an effective config to `.llm_perf_effective.yaml`.
2. `llm_perf.locustfile` reads `LLM_PERF_CONFIG`, builds one Locust `User` class per
   scenario via `scenarios.build_users()`.
3. Each virtual user runs a `TaskSet` that calls `RequestContext.run_scenario()`.
4. `RequestContext` resolves the target through `LayerFactory`, picks a prompt or
   dataset row, merges scenario params, and calls `layer.run()`.
5. The layer executes the request:
   - `llm_api` → `providers.call_llm()` → provider adapter → HTTP/gRPC/boto3
   - `pipeline` → `pipelines.run_pipeline()` → stage engine (retrieve/rerank/generate/agent)
   - `app_endpoint` → `transports/http.py` or `transports/grpc.py`
   - `app_object` → dynamic import + async invocation
6. On completion the layer records:
   - an OTel root span (`llm_perf.request`) with GenAI attributes
   - OTel metrics (`llm_perf.*` instruments)
   - Locust request stats + custom stats (TTFT, TPOT, ITL, etc.)
   - stage spans/metrics for pipelines
   - cost and agent-eval counters
7. After quality evaluation, `tasks.py` records quality metrics and cost.
8. At `test_stop`, `llm_perf.locustfile` aggregates stats, evaluates thresholds,
   enriches the report with phases/stages/resources/analysis/comparison, writes
   CSV/JSON/HTML, records history, sends alerts, and sets the exit-code flag.

## Concurrency model

Locust runs virtual users as **gevent greenlets** on a single OS thread. All
provider I/O is monkey-patched and yields the hub, so hundreds of users share
one thread cooperatively. CPU-bound work does not yield, which is why the
framework ships a scheduler-lag probe and generator-health warnings: high lag
or CPU/memory saturation inflates streaming timestamps (TTFT/TPOT/ITL).

See [methodology.md](methodology.md) for details and remedies (reduce users,
shard with `--master/--worker`, bigger host).

## Observability pipeline

```
llm-perf process
   │ OTLP/gRPC (traces + metrics)
   ▼
OTel Collector (otel-collector-config.yaml)
   ├── Prometheus exporter → Prometheus (:8889 scrape → :9090)
   └── Jaeger exporter     → Jaeger UI (:16686)
Grafana (:3000) ← provisioned dashboard (grafana/llm-perf-dashboard.json)
```

- Traces: one root span per request plus child spans per pipeline stage/tool call.
- Metrics: `llm_perf.*` instruments exported as Prometheus metrics.
- Threshold gauges: `llm_perf_threshold_pass` is set live during threshold evaluation.

## Reporting pipeline

`llm_perf/locustfile.py::_run_reports_and_thresholds` does the post-run work:

1. `reporters.aggregate()` builds the base report from Locust stats + exact custom samples.
2. `_enrich_report()` adds `load`, `phases`, `stages`, `resources`, `warnings`,
   `analysis` (executive summary, SLO rollup, bottlenecks), `comparison`, `regression`.
3. `reporters.write_reports()` writes CSV/JSON/HTML into `report.output_dir`.
4. `history.record_run()` archives the JSON for future baselines.
5. `thresholds.send_alerts()` posts to configured webhooks.
6. `LLM_PERF_THRESHOLDS_FAILED=1` is set if thresholds or regressions fail, so the
   CLI can exit non-zero for CI.

## Layer responsibilities

| Layer | Measures | Does not measure |
|---|---|---|
| `llm_api` | Provider latency, TTFT, tokens/sec, token counts, cost, retries/timeouts | Application logic around the model |
| `pipeline` | End-to-end pipeline latency, per-stage latency, context recall, agent steps/tool calls/loops/success, token/cost rollup | Direct provider internals (uses providers underneath) |
| `app_endpoint` | Black-box HTTP/gRPC latency, error rate, OTel trace correlation | Token usage or model quality (unless the app exposes them) |
| `app_object` | In-process framework/object latency, output quality | Network/transport overhead |

## Key design choices

- **Exact samples + Locust stats**: Locust `StatsEntry` buckets percentiles for the
  live UI; `metrics.py` keeps full-precision samples for reports and thresholds.
- **Declarative config**: everything is YAML; code only executes what the config describes.
- **Provider abstraction**: `RequestSpec` → `NormalizedResponse` keeps metrics uniform
  across OpenAI/Anthropic/Azure/Bedrock/HTTP/gRPC.
- **Quality as a first-class signal**: heuristics are free and always on; LLM judge is
  sampled; RAG recall is computed when expected context exists.
- **Baseline/regression tracking**: every run is archived; reports can compare against
  previous/designated baselines and flag regressions.
