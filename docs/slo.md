# Availability and latency SLOs

`llm-perf` can define and enforce benchmark SLOs for request availability,
functional availability, non-degraded availability, latency percentiles,
throughput, and error rate.

The feature is designed for load-test evidence and CI gates. It is not a
replacement for a production SLA platform that tracks rolling 30-day uptime,
error budgets, and contractual exclusions.

## What each availability metric means

Every completed `llm-perf` task records one request outcome:

| Metric | Definition | Typical meaning |
|---|---|---|
| `availability` | Request completed without a layer error | Basic request success rate (`1 - error_rate`) |
| `functional_availability` | Request completed **and** configured functional checks passed | User received a usable answer, not merely HTTP success |
| `non_degraded_availability` | Request completed **without** resilience fallback/degradation | Primary dependency path served the request |

Functional checks currently include:

- `scenario.quality` pass/fail when quality evaluation is configured
- agent task success when the pipeline emits an agent evaluation

Missing checks do not reduce functional availability. For example, a plain
`llm_api` request with no quality scorer has `functional_availability ==
availability`; a fallback-completed request can remain functionally available
but is not non-degraded.

## Configuration

SLOs are normal threshold rules. They can be global or per scenario:

```yaml
thresholds:
  global:
    min_availability: 0.99
    max_error_rate: 0.01

  scenarios:
    rag_load:
      min_availability: 0.995
      min_functional_availability: 0.95
      min_non_degraded_availability: 0.90
      p50_ms: 1500
      p95_ms: 8000
      p99_ms: 15000
      max_error_rate: 0.005

    agent_load:
      min_availability: 0.98
      min_functional_availability: 0.85
      min_non_degraded_availability: 0.90
      p95_ms: 30000
      min_agent_success_rate: 0.80

  on_fail: fail
  on_not_evaluated: warn
```

Use `on_fail: fail` to make a breach return a non-zero `llm-perf run` exit
code. Use `on_not_evaluated: fail` in strict CI when missing metric data should
also fail the run.

## How it works

At the end of every request, `RequestContext` records:

1. Whether the layer returned an error.
2. Whether the configured quality check passed.
3. Whether the agent evaluation marked the task successful.
4. Whether any resilience-wrapped dependency call produced degraded output.
5. The request end-to-end duration and timestamp.

Whole-run availability rates are stored in uncapped request counters. A second,
bounded timestamped sample store powers phase/window summaries so spike and
soak reports can show short outage windows without retaining unlimited request
history in memory.

## Phase SLO windows

When the workload produces a phase schedule, reports include `slo_windows`.
Each phase row contains the scenario request count, the three availability
rates, and p50/p95/p99 latency for that phase.

This helps answer questions such as:

- Did availability drop only during the spike phase?
- Did the service recover during the post-outage window?
- Was a fallback used during one phase, lowering non-degraded availability but
  preserving functional availability?

For deterministic baseline → outage → recovery windows, combine SLO windows
with `resilience.fault_injection`.

## Report fields

Scenario rows in CSV/JSON/HTML include:

- `availability`
- `functional_availability`
- `non_degraded_availability`

Report totals include the weighted cross-scenario values. The HTML report
shows the same metrics in summary cards, the scenario table, and the `SLO
windows` table when phases exist.

Threshold evaluation uses the same request-outcome data. If no request SLO
sample exists, `min_availability` falls back to the Locust request
success/failure ratio; functional and non-degraded rules are `not_evaluated`
unless request-outcome data exists.

## How to run

Validate and execute a normal run:

```bash
llm-perf validate -c configs/my_pipeline.yaml
llm-perf run -c configs/my_pipeline.yaml --headless
```

For a resilience-aware SLO test using the checked-in example:

```bash
llm-perf validate -c configs/rag_resilience.yaml --skip-connectivity
llm-perf run -c configs/rag_resilience.yaml --headless
```

Then inspect `report.output_dir`:

- `llm-perf-*.json` — full scenario fields, thresholds, and `slo_windows`
- `llm-perf-*.csv` — one row per scenario
- `llm-perf-*.html` — human-readable SLO evidence

## Boundaries

- These are benchmark-window SLOs, not calendar SLAs.
- There is no rolling 5-minute/30-day error budget or burn-rate alert yet.
- `availability` means the `llm-perf` layer completed; use
  `min_functional_availability` when output quality or agent success matters.
- External black-box services can only expose observable request outcomes. To
  classify their internal degraded state, export service-side telemetry and
  compare it with this report.
