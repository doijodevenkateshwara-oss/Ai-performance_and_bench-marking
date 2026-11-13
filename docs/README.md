# llm-perf documentation

This directory contains the framework documentation:

| Document | What it covers |
|---|---|
| [framework.md](framework.md) | File-by-file guide: what every file and module does |
| [architecture.md](architecture.md) | System architecture, request lifecycle, layers, observability, reporting |
| [how-to-run.md](how-to-run.md) | Install, configuration, CLI usage, profiles, Docker stack, CI |
| [test-results.md](test-results.md) | Report formats, sample results, metric definitions, thresholds, baselines, interpretation |
| [llm-api-performance-testing.md](llm-api-performance-testing.md) | End-to-end direct-LLM benchmarking: providers, streaming metrics, config, running, results, validation, CI flow |
| [rag-performance-testing.md](rag-performance-testing.md) | End-to-end RAG benchmarking: pipeline config, datasets, running, results, validation, CI flow |
| [app-endpoint-performance-testing.md](app-endpoint-performance-testing.md) | Black-box app endpoint benchmarking: HTTP/gRPC config, templates, datasets, running, results, validation, CI flow |
| [app-object-performance-testing.md](app-object-performance-testing.md) | In-process LangChain/LlamaIndex/custom-object benchmarking: python_path/call contract, config, running, results, validation, CI flow |
| [workload-profiles.md](workload-profiles.md) | Load shaping: nine workload profiles, custom stages, think time, E2E test flow, distributed runs, result interpretation |
| [methodology.md](methodology.md) | Timing accuracy, gevent concurrency model, when streaming metrics are unreliable |
| [demo.md](demo.md) | Live demo: start the Docker observability stack, dashboard URLs, run a mock test |

Dashboard screenshots live in [screenshots/](screenshots/):

| File | Dashboard |
|---|---|
| `screenshots/grafana-llm-perf.png` | Grafana — full LLM Performance & Quality dashboard (all 10 panels) |
| `screenshots/prometheus-graph.png` | Prometheus — p95 request-duration graph |
| `screenshots/prometheus-targets.png` | Prometheus — scrape targets/status |
| `screenshots/jaeger-search.png` | Jaeger — trace search results for `llm-perf-demo` |
| `screenshots/jaeger-trace.png` | Jaeger — individual trace detail (span waterfall) |
| `screenshots/locust-ui.png` | Locust live UI during a run |

These are committed captures of a live demo run (see [demo.md](demo.md) to
reproduce the stack and generate fresh ones).
