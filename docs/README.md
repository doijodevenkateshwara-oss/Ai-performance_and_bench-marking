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
| `screenshots/grafana-llm-perf.png` | Grafana — LLM Performance & Quality dashboard |
| `screenshots/prometheus.png` | Prometheus — p95 request-duration graph |
| `screenshots/jaeger-traces.png` | Jaeger — trace search for `llm-perf-demo` |
| `screenshots/locust-ui.png` | Locust live UI during a run |

Regenerate the Docker-stack shots with
`.venv/bin/python scripts/screenshot_dashboards.py` (stack must be running via
`docker compose up -d`) and the Locust shot with
`.venv/bin/python scripts/screenshot_dashboard.py`. See [demo.md](demo.md).
