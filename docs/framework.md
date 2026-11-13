# Framework file guide

`llm-perf` is a Python package plus a set of YAML configs, datasets, scripts,
and observability assets. This page maps every file/directory to its role.

## Repository root

| Path | Purpose |
|---|---|
| `README.md` | Project overview, quick start, layer/provider reference, metrics catalog, CI notes |
| `pyproject.toml` | Package metadata, dependencies, optional extras (`dev`, `langchain`, `llamaindex`, `s3`, `gcs`), entry point `llm-perf = llm_perf.cli:app`, ruff/black/pytest config |
| `locustfile.py` | Thin shim so the standard `locust -f locustfile.py` CLI can import the generated `User`/`LoadTestShape` classes from `llm_perf.locustfile` |
| `.env.example` | Template for secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, AWS creds, webhook URLs) |
| `.gitignore` | Ignores `.venv/`, `__pycache__/`, `.env`, caches, `*.egg-info/` |
| `docker-compose.yml` | Root shim (`include:`) so `docker compose up -d` works from the repo root |
| `deployments/docker/docker-compose.yml` | Local observability stack: OTel collector, Prometheus, Jaeger, Grafana |
| `deployments/prometheus/prometheus.yml` | Prometheus scrape config — pulls `llm_perf_*` metrics from the OTel collector on `:8889` |
| `deployments/otel/collector-config.yaml` | OTel Collector config: OTLP gRPC/HTTP in → Prometheus exporter + Jaeger exporter |

## `llm_perf/` package

| Path | Purpose |
|---|---|
| `llm_perf/__init__.py` | Package marker |
| `llm_perf/cli.py` | Typer CLI: `run`, `validate`, `proto-gen`, `quality-check`, `dataset list/show/sample/import/validate/coverage`, `baseline mark/list` |
| `llm_perf/config.py` | Pydantic schema for the whole YAML config: observability, security, datasets, targets (4 layers), scenarios, load, thresholds, report |
| `llm_perf/locustfile.py` | Locust entry point: reads `LLM_PERF_CONFIG`, builds weighted `User` classes, optional `LLMPerfShape`, wires `test_start`/`test_stop`/`quitting` hooks for OTel, resource sampling, reports, thresholds, alerts |
| `llm_perf/scenarios.py` | Turns `scenarios[]` into one Locust `User` class per scenario, with `wait_time` from `think_time` and weighted selection |
| `llm_perf/tasks.py` | `RequestContext`: per-scenario runtime state — resolves the layer, picks prompt/dataset row, runs the request, evaluates quality, accumulates cost |
| `llm_perf/metrics.py` | OTel instruments + Locust custom-stat bridge; exact sample store for TTFT/TPOT/ITL/gen-duration/queue-wait/retries/timeouts/quality/context-recall/IR (recall@k/precision@k/MRR/NDCG)/agent metrics; per-request latency breakdown for percentile-safe composition; cost-by-call-site accumulation; `request_span`, `record_response`, `record_error`, `record_stage`, `record_cost`, `record_agent_eval`, `record_users`, `record_threshold`, `estimate_cost` |
| `llm_perf/security.py` | Egress/execution guardrails: URL allowlist + private-network (SSRF) checks on all outbound HTTP, Python-execution gate for `python_path`/function tools, secret redaction for error stats/spans, untrusted-config audit warnings |
| `llm_perf/reporters.py` | Aggregates Locust stats into a report dict and writes CSV/JSON/HTML under `report.output_dir` |
| `llm_perf/thresholds.py` | SLO rule evaluation (`ThresholdRule` → `RuleResult`), threshold gauges, webhook alerts (Slack/PagerDuty/generic) |
| `llm_perf/analysis.py` | Post-run analysis: executive summary, SLO rollup, bottleneck detection, baseline comparison, regression detection |
| `llm_perf/history.py` | Persists every run under `report.output_dir/history/`, maintains `index.jsonl`, designated/previous baselines |
| `llm_perf/resources.py` | Load-generator host sampler (psutil CPU/mem), event-loop lag probe, Prometheus `/metrics` scraper for target-side metrics, generator-health warnings |
| `llm_perf/workload.py` | Translates `load.profile`/`stages` into phase schedules (`Phase` list) consumed by the Locust `LoadTestShape` |
| `llm_perf/datasets.py` | Loads YAML/CSV/JSONL datasets from local/HTTP/S3/GCS, column mapping, sampling, `{var}` templating, deterministic sharding + `validate_shards` coverage check for distributed workers, trace import |
| `llm_perf/prompts.py` | Loads prompt pools from inline lists or YAML/CSV/TXT files, `PromptPicker` with row/vars `{var}` rendering |
| `llm_perf/streaming.py` | Re-exports SSE helpers (`stream_sse`, `stream_timing_stats`, OpenAI/Anthropic extractors) used by providers |
| `llm_perf/proto.py` | CLI helper for compiling `.proto` files for gRPC targets |
| `llm_perf/templates/report.html` | Jinja2 template for the HTML report (executive summary, scenario table, charts, bottleneck/SLO/cost/resource sections) |

### `llm_perf/otel/` — observability

| Path | Purpose |
|---|---|
| `otel/setup.py` | Initializes OTel `TracerProvider`/`MeterProvider` with OTLP/gRPC export, configurable endpoint + sample ratio, `get_tracer`/`get_meter`/`shutdown` |
| `otel/genai_semconv.py` | Creates all `llm_perf.*` instruments (request duration/total, TTFT, tokens, tokens/sec, TPOT, generation duration, ITL, queue wait, retries, timeouts, users, stage duration, quality, context recall, cost, threshold pass, agent metrics) and GenAI semantic-convention attribute names |

### `llm_perf/providers/` — direct LLM API adapters

| Path | Purpose |
|---|---|
| `providers/base.py` | `NormalizedResponse`, `RequestSpec`, `ProviderAdapter` protocol, `usage_tokens` helper |
| `providers/__init__.py` | Adapter registry, `call_llm` entry point, retry/backoff wrapper for HTTP providers, queue-wait capture |
| `providers/http_common.py` | Shared HTTP/SSE machinery: `HttpClient`, `stream_sse`, `stream_timing_stats`, OpenAI/Anthropic SSE extractors |
| `providers/openai.py` | OpenAI chat-completions + Responses API adapter, streaming + token usage via `stream_options.include_usage` |
| `providers/azure.py` | Azure OpenAI adapter — deployment URL routing, `api-key` header, `api-version` query param |
| `providers/anthropic.py` | Anthropic Messages API adapter, streaming via SSE event types |
| `providers/bedrock.py` | AWS Bedrock Converse API adapter via boto3 in a thread pool, unary + streaming |
| `providers/gemini.py` | Google Gemini generateContent/streamGenerateContent adapter — `x-goog-api-key` auth, generationConfig mapping, usageMetadata → token counts, Vertex AI via `{model}` endpoint override |
| `providers/http.py` | Generic HTTP provider for OpenAI-ish JSON endpoints, optional Jinja request template |
| `providers/grpc.py` | gRPC adapter: compiles `.proto`, loads stubs, builds request messages, calls Triton-style RPCs |

### `llm_perf/transports/` — app endpoint transports

| Path | Purpose |
|---|---|
| `transports/http.py` | `HttpTransport` — thin `requests` wrapper with `call_timed` for the `app_endpoint` layer |
| `transports/grpc.py` | `GrpcTransport` — reuses `providers/grpc.py` for black-box gRPC endpoints |

### `llm_perf/layers/` — target implementations

| Path | Purpose |
|---|---|
| `layers/base.py` | `Layer` protocol and `LayerResult` dataclass |
| `layers/__init__.py` | `LayerFactory`: caches layers/datasets/prompt pickers, builds the right layer for a target, handles worker sharding |
| `layers/llm_api.py` | Layer 1 — direct vendor API calls via `providers.call_llm`, records metrics/traces/cost |
| `layers/pipeline.py` | Layer 2 — declarative RAG/agent pipelines via `pipelines.run_pipeline`, records stage metrics, agent eval, usage |
| `layers/app_endpoint.py` | Layer 3 — black-box HTTP/gRPC app endpoint, templated requests, pure transport latency/error rate |
| `layers/app_object.py` | Layer 4 — in-process LangChain/LlamaIndex/generic Python object, dynamic import + async call, best-effort text extraction; also exports `langchain_result_text`/`llamaindex_result_text` helpers |

### `llm_perf/pipelines/` — declarative pipeline engine

| Path | Purpose |
|---|---|
| `pipelines/engine.py` | Executes `retrieve`/`rerank`/`generate`/`agent` stages, builds `StageContext`, records per-stage spans/metrics, implements the OpenAI-compatible tool-calling agent loop with loop detection and task-success evaluation |
| `pipelines/__init__.py` | Re-exports `StageContext`, `run_pipeline` |

### `llm_perf/quality/` — quality scoring

| Path | Purpose |
|---|---|
| `quality/__init__.py` | Re-exports quality API |
| `quality/heuristics.py` | Zero-cost checks: non-empty, min/max chars, required/forbidden regex, contains, JSON-valid, exact-match field, refusal detection |
| `quality/judge.py` | LLM-as-judge scorer via any provider, `JudgeSampler` for `sample_rate` control, score parsing/normalization |
| `quality/rag.py` | RAG context recall — token-overlap recall of retrieved text vs dataset `expected_context` |
| `quality/scorer.py` | `evaluate_quality`: combines heuristics + judge + context recall into a 0..1 score, pass/fail, failure tags |

## Configs, data, prompts, protos

| Path | Purpose |
|---|---|
| `configs/openai_example.yaml` | OpenAI streaming chat + dataset + quality/judge example |
| `configs/anthropic_example.yaml` | Anthropic Messages API example |
| `configs/azure_example.yaml` | Azure OpenAI deployment example |
| `configs/bedrock_example.yaml` | AWS Bedrock Converse API example |
| `configs/grpc_example.yaml` | gRPC/Triton inference example |
| `configs/rag_example.yaml` | RAG pipeline example (retrieve → rerank → generate) |
| `configs/agent_example.yaml` | Agent pipeline example (tool-calling loop) |
| `configs/profiles_example.yaml` | Workload profiles, retries, rich reporting/history/regression example |
| `datasets/qa_pairs.jsonl` | Sample QA pairs with expected answers/context for quality and RAG recall |
| `datasets/docs.jsonl` | Local JSON document store used by the RAG example's `retrieve` stage |
| `prompts/rag_answer.j2` | Jinja2 template for the RAG generate stage |
| `protos/inference.proto` | Minimal Triton/KServe-style `ModelInfer`/`ModelStreamInfer` gRPC proto |

## Tests

Tests are organized by scope: `tests/unit/` (mocked/local, no services),
`tests/integration/` (real local HTTP servers), `tests/e2e/` (subprocess runs).

| Path | Purpose |
|---|---|
| `tests/unit/test_config.py` | Validates example configs and config normalization/rejection rules |
| `tests/unit/test_datasets.py` | Dataset loading, column mapping, sampling, templating, sharding, trace import/export |
| `tests/unit/test_prompts.py` | Prompt file loading and `{var}` rendering/row priority |
| `tests/unit/test_providers.py` | Provider request building/parsing for OpenAI, Anthropic, Azure, usage extraction |
| `tests/unit/test_streaming.py` | SSE parsing, TTFT/TPOT/ITL/generation-duration derivation |
| `tests/unit/test_evaluation.py` | Heuristic checks, judge score parsing, quality combination |
| `tests/unit/test_thresholds.py` | SLO rule evaluation with fake Locust stats |
| `tests/unit/test_security.py` | URL allowlist/private-network guards, secret redaction, python-execution gate, dataset limits, config audit |
| `tests/unit/test_reports.py` | Report aggregation and CSV/JSON/HTML writing |
| `tests/unit/test_agent_eval.py` | Agent-stage success, tool correctness, loop detection, cost cap |
| `tests/unit/test_grpc.py` | Bundled proto compilation and stub loading |
| `tests/unit/test_workstreams_c.py` | Workload profiles, analysis/bottleneck/regression/history, resource sampling |
| `tests/integration/test_prometheus.py` | Prometheus `/metrics` scraping against a real local HTTP server |
| `tests/e2e/test_locust_execution.py` | Real headless Locust run against a mock OpenAI server + direct layer smoke test |
| `tests/e2e/e2e_runner.py` | Subprocess runner used by `test_locust_execution.py` (Locust must run in its own process) |
| `tests/unit/agent_tools.py` | Helper tool functions used by agent-eval tests |

## Observability assets

| Path | Purpose |
|---|---|
| `deployments/grafana/dashboards/llm-perf-dashboard.json` | Provisioned Grafana dashboard: request rate, error rate, e2e latency percentiles, TTFT, tokens/sec, quality, active users, cost, stage latency, threshold status |
| `deployments/grafana/provisioning/datasources/prometheus.yml` | Grafana datasource pointing at Prometheus |
| `deployments/grafana/provisioning/dashboards/dashboards.yml` | Grafana dashboard provider config |

## Generated output (not source)

| Path | Purpose |
|---|---|
| `results/` | Reports from local runs (`llm-perf-*.csv/json/html`) plus `history/` archives |
| `docs/sample-results/` | Published example reports from a mock OpenAI streaming run |
| `docs/screenshots/` | Dashboard/UI screenshots (Grafana, Prometheus, Jaeger, Locust) |
| `llm_perf.egg-info/` | Setuptools build metadata from `pip install -e .` |
