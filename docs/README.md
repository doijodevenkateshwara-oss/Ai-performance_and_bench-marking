# llm-perf documentation

This directory contains the framework documentation:

| Document | What it covers |
|---|---|
| [framework.md](framework.md) | File-by-file guide: what every file and module does |
| [architecture.md](architecture.md) | System architecture, request lifecycle, layers, observability, reporting |
| [how-to-run.md](how-to-run.md) | Install, configuration, CLI usage, profiles, Docker stack, CI |
| [test-results.md](test-results.md) | Report formats, sample results, metric definitions, thresholds, baselines, interpretation |
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

Regenerate the Docker-stack shots with
`.venv/bin/python scripts/screenshot_dashboards.py` (stack must be running via
`docker compose up -d`) and the Locust shot with
`.venv/bin/python scripts/screenshot_dashboard.py`. See [demo.md](demo.md).
