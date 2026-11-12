# Understanding test results

`llm-perf` produces three report formats plus live telemetry. This page
explains what each artifact contains and how to interpret it.

## Report artifacts

Every run writes files named `llm-perf-<timestamp>.{csv,json,html}` into
`report.output_dir` (default `results/`).

| Format | Best for | Contents |
|---|---|---|
| CSV | Spreadsheets, quick diffs | One row per scenario with requests, failures, error rate, latency percentiles, TTFT/TPOT/ITL/gen-duration/queue-wait, tokens/sec, retries/timeouts, quality, context recall, cost, agent metrics |
| JSON | CI, regression tracking, programmatic analysis | Full report dict: totals, per-scenario methods + enriched metrics, thresholds verdicts, load profile, phases, stages, resources, analysis, comparison, regression |
| HTML | Stakeholders, executive review | Verdict banner, summary cards, per-scenario table, latency composition chart, stage breakdown, bottleneck analysis, SLO rollup, cost, resource utilization, baseline comparison, regression detection, quality |

## Sample results

`docs/sample-results/` contains a real generated run against a mock OpenAI
streaming server (no credentials required).

Example JSON highlights (`llm-perf-20261112-185115.json`):

- **Run shape**: `smoke` profile, 3 users, ~30 s, ramp-up 5 s → steady 20 s → ramp-down 5 s
- **Totals**: 52 requests, 0 failures, 0% error rate, peak 3 users
- **Headline scenario**: `stream_smoke`
- **Latency**: p50/p90/p95/p99 ≈ 370 ms end-to-end
- **Streaming split**:
  - TTFT p95 ≈ 90.9 ms (request sent → first token)
  - TPOT p95 ≈ 24.8 ms (decode cadence per output token)
  - Generation duration p95 ≈ 273 ms (first → last token)
  - ITL p95 ≈ 91.6 ms (chunk-to-chunk gap)
- **Throughput**: ~44.2 output tokens/s, ~32.9 input tokens/s
- **Queue wait**: p95 ≈ 0.10 ms (negligible scheduling overhead)
- **Retries/timeouts**: 0 / 0
- **Cost**: `null` — no `report.pricing` was configured for the mock run
- **Quality**: no quality rules configured → `null`
- **Analysis verdict**: `PASS`; bottleneck flagged as `decode_bound` (generation p95 dominates e2e)
- **Resources**: load generator CPU avg 17.8%, mem max 27.2%, scheduler lag p95 1.04 ms — healthy

## Key metrics

All times are milliseconds in reports; seconds internally.

| Metric | Meaning | Why it matters |
|---|---|---|
| `requests` / `failures` / `error_rate` | Completed requests, failures, failure ratio | Basic reliability SLO |
| `avg_ms`, `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms` | End-to-end request latency percentiles | Overall user-perceived latency |
| `rps` | Requests per second | Throughput capacity |
| `ttft_avg_ms` / `ttft_p95_ms` | Time to first token | Prefill/queueing on the provider; critical for chat UX |
| `tpot_avg_ms` / `tpot_p95_ms` | Time per output token | Decode speed; lower is better for long responses |
| `gen_duration_avg_ms` / `gen_duration_p95_ms` | First token → last token | Total decode window |
| `itl_avg_ms` / `itl_p95_ms` | Inter-token latency | Chunk cadence; spikes indicate jitter or generator lag |
| `input_tokens_per_sec_avg` / `tokens_per_sec_avg` | Prompt and output throughput | Cost/performance efficiency |
| `queue_wait_avg_ms` / `queue_wait_p95_ms` | Task start → request dispatch | Locust think-time/scheduling overhead; high values suggest generator saturation |
| `retries_total` / `timeouts` | Retry attempts and timed-out requests | Reliability under flaky networks/providers |
| `total_tokens` | Sum of prompt + completion tokens | Usage volume |
| `quality_score_avg` | 0..1 quality score (heuristics + judge) | Correctness/helpfulness under load |
| `quality_pass_rate` | Fraction of quality checks passed | Binary quality gate |
| `context_recall_avg` | RAG context recall vs expected context | Retrieval quality |
| `cost_usd` / `cost_per_success_usd` | Estimated spend and spend per successful request | Cost efficiency |
| `agent_runs` / `agent_success_rate` | Agent task runs and success rate | Agent reliability |
| `agent_steps_avg` / `agent_tool_calls_avg` / `agent_tool_errors_avg` / `agent_loops_avg` | Agent loop behavior | Efficiency and failure modes |

## Thresholds & SLOs

Threshold rules live under `thresholds:` in the YAML config. They can be
`global` or per-`scenario`, and include latency percentiles, custom p95 rules
(TTFT/TPOT/ITL/gen/queue), gauges (error rate, throughput, quality, cost,
agent metrics), and per-`stage` rules for pipelines.

Example verdicts in the JSON report:

```json
"thresholds": {
  "all_passed": false,
  "failed": ["chat_load.p95_ms"],
  "rules": [
    {
      "rule": "chat_load.p95_ms",
      "metric": "p95_latency_ms",
      "comparator": "<=",
      "threshold": 30000,
      "actual": 45210.5,
      "passed": false,
      "status": "fail",
      "scope": "scenario"
    }
  ]
}
```

Every rule has a three-state `status`: `pass`, `fail`, or `not_evaluated`
(no data for that metric — e.g. a TTFT rule on a non-streaming run).

- `on_fail: fail` → CLI exits non-zero on threshold failures (CI gate)
- `on_fail: warn` → failures are reported but exit code stays 0
- `on_not_evaluated: pass` (default) → missing-data rules count as pass
- `on_not_evaluated: warn` → unevaluated rules print a warning, exit stays 0
- `on_not_evaluated: fail` → unevaluated rules exit non-zero like failures
- `llm_perf_threshold_pass` gauge is exported live for Grafana/Alertmanager

## Analysis sections

The JSON/HTML reports include an `analysis` object:

- **executive_summary** — verdict (`PASS`/`WARN`/`FAIL`/`INCOMPLETE`/`NO DATA`),
  headline scenario, top-line findings, SLO pass/fail/not_evaluated counts
- **slo** — rollup of threshold rules into pass/fail/not_evaluated counts
  (`skip` kept as a backward-compatible alias for `not_evaluated`)
- **bottlenecks** — ranked findings such as:
  - `stage` — a pipeline stage dominates e2e latency
  - `ttft_bound` — TTFT is >60% of e2e p95 (prefill/queue-bound)
  - `decode_bound` — generation duration dominates e2e (decode-bound)
  - `reliability` — high retries/timeouts
  - `queueing` — queue wait is a large share of e2e (load-generator saturation)

## History, baselines & regression

If `report.history: true` (default), every run is archived under
`report.output_dir/history/` as `run-<timestamp>.json` with an `index.jsonl`.

Compare against:

- `baseline: previous` — most recent prior run
- `baseline: designated` — the run marked by `llm-perf baseline mark`
- `baseline: /path/to/run.json` — explicit file

`analysis.compare_runs` produces per-metric deltas (`current`, `baseline`,
`delta`, `delta_pct`, `direction`). `analysis.detect_regressions` flags metrics
that moved worse than allowed percentages (`report.regression.thresholds_pct`,
with defaults for p95/p99/TTFT/TPOT/ITL/gen/queue/error-rate/retries/timeouts/
cost/quality/tokens-s/RPS).

`report.regression.on_fail: fail` turns regressions into a CI gate.

## Resource utilization & generator health

`report.collect_resources: true` samples the load generator with psutil:

- `resources.loadgen.cpu_percent_avg/max`
- `resources.loadgen.mem_percent_avg/max`, `mem_used_mb_max`
- `resources.loadgen.scheduler_lag` — event-loop lag probe stats

`report.scrape` can also pull target-side Prometheus metrics into
`resources.targets` for comparison.

`resources.py::evaluate_resource_warnings` compares these against
`report.resource_warn` thresholds and adds `warnings` to the report/HTML when
the generator is saturated. High lag means streaming timestamps are
scheduling-inclusive and should be treated as unreliable.

## Quality signals

Quality is evaluated per scenario when `quality:` is configured:

- **heuristics** — always free: non-empty, min/max chars, regex patterns,
  JSON validity, exact match, refusal detection
- **judge** — LLM-as-judge on a `sample_rate` fraction of responses
- **rag** — context recall when `expected_context` exists in the dataset

The combined `quality_score_avg` (0..1) and `quality_pass_rate` appear beside
performance metrics, so you can see whether quality holds up under load.

## Cost

Set `report.pricing.input_per_1k` and `report.pricing.output_per_1k` (USD per
1K tokens) to get:

- `llm_perf.cost.USD` OTel counter
- `cost_usd` per scenario and `cost_per_success_usd`
- `max_cost` / `max_cost_per_success_usd` threshold rules

Cost is an estimate from token counts and configured pricing, not billing data.

## Agent evaluation

For pipeline targets with an `agent` stage, the report adds:

- `agent_runs`, `agent_success_rate`
- `agent_steps_avg`, `agent_tool_calls_avg`, `agent_tool_errors_avg`,
  `agent_tool_error_rate`, `agent_loops_avg`
- termination reasons (`final_answer`, `max_iterations`, `cost_cap`, `error`)

These help distinguish "the model is slow" from "the agent is looping",
"tools are failing", or "the task never succeeds".

## Interpreting bottlenecks

Use the `analysis.bottlenecks` list as the first stop:

- **TTFT-bound**: prefill/queueing dominates → look at provider/model, prompt
  length, concurrency, or provider-side queuing.
- **Decode-bound**: generation duration dominates → look at max_tokens, model
  decode speed, streaming chunk cadence.
- **Stage-bound**: a pipeline stage (retrieve/rerank/generate/agent_step/
  agent_tool) dominates → optimize that stage or its backing service.
- **Reliability**: retries/timeouts dominate → check provider health, network,
  timeout settings.
- **Queueing**: queue wait is high → the load generator is saturated; reduce
  users, shard, or use a bigger host.

## Live dashboards

While a run is active:

- **Locust UI** (`:8089`) — RPS, latency percentiles, failures, custom stats
- **Grafana** (`:3000`) — `llm-perf` dashboard with request rate, error rate,
  latency percentiles, TTFT, tokens/sec, quality, active users, cost, stage
  latency, threshold status
- **Prometheus** (`:9090`) — raw `llm_perf_*` metrics
- **Jaeger** (`:16686`) — per-request traces with stage spans

After the run, the HTML report is the best single artifact for sharing.
