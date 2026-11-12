# Methodology & timing accuracy

How llm-perf generates load, measures streaming latency, and how to interpret
the numbers — including when they are *not* trustworthy.

## Concurrency model: gevent greenlets (Locust default)

Locust runs each virtual user as a **gevent greenlet** on a single OS thread
(one event-loop "hub"). All provider calls (`requests`, `boto3`, `grpc`)
are blocking socket operations that gevent monkey-patches, so they yield the
hub while waiting on I/O. That is how hundreds/thousands of "concurrent"
users share one thread: concurrency is cooperative, not OS-threaded.

Consequences:

- **One greenlet at a time runs Python code.** While user A's callback is
  executing, users B..N are frozen. I/O waits yield; CPU work does not.
- **Streaming reads yield per socket read.** `iter_lines()` on an SSE
  response suspends the greenlet until the next chunk arrives, so chunk
  timestamps are taken in wall-clock time but are *scheduling-inclusive*:
  if the hub is busy running other greenlets, a chunk's recorded arrival
  time is later than its true arrival time.
- **asyncio is not used.** llm-perf does not run an asyncio event loop per
  user; the whole framework is gevent-based via Locust. If your application
  under test is asyncio-based that does not matter — only the *generator*
  is gevent.

### When TTFT / ITL / TPOT become unreliable

TTFT, TPOT, ITL, and generation duration are measured from chunk timestamps.
Under high concurrency or a busy generator, greenlet scheduling delay is
added to every inter-chunk gap, inflating TPOT/ITL and sometimes TTFT.

Signals that your generator is the bottleneck, not the target:

- `resources.loadgen.scheduler_lag.p95_ms` in the JSON report is high
  (default warn threshold: `report.resource_warn.lag_p95_ms: 50`).
- `cpu_percent_avg` > 85 or `mem_percent_max` > 90 on the load generator.
- The HTML report shows a **Generator-health warnings** banner with the
  same checks.

Remedies:

- Reduce `users`/concurrency, or shorten `spawn_rate`.
- Shard the run: `locust --master` + N `--worker` processes (each worker
  is a separate gevent hub), then aggregate.
- Run the generator on a bigger host or closer to the target (network RTT
  also lands inside TTFT/ITL).

## Metric definitions

All times are seconds internally; reports emit milliseconds.

| Metric | Definition |
| --- | --- |
| TTFT | dispatch → first streamed content chunk (non-streaming: full latency) |
| TPOT | mean inter-chunk interval during decode, `(gen_duration) / (chunks-1)` |
| ITL | per-chunk inter-arrival times; reports `itl_avg_ms` and `itl_p95_ms` |
| Generation duration | first chunk → last chunk (decode window; excludes TTFT) |
| End-to-end latency | full request wall time, queue wait excluded |
| Queue wait | task start → request dispatch (Locust think-time/scheduling overhead) |
| Input / output tok/s | prompt tokens / wall time, completion tokens / generation duration |
| RPS | completed requests / run duration |
| Cost per success | total estimated cost / (requests − failures) |

These are deliberately distinct: **TTFT** is prefill+queueing on the
provider, **TPOT/ITL** is decode cadence, **generation duration** is the
whole decode window, and **end-to-end** adds network + post-processing.
A regression in one does not imply the others moved.

## Event-loop lag probe

`llm_perf.resources.LagProbe` is a greenlet that sleeps a fixed interval
(250 ms) and measures overshoot — how late `sleep()` actually wakes. That
directly measures hub scheduling delay, which is exactly what inflates
streaming timestamps. Summary stats (`avg_ms`, `p95_ms`, `max_ms`) land in
`resources.loadgen.scheduler_lag` and feed `evaluate_resource_warnings`.

Configure thresholds:

```yaml
report:
  resource_warn:
    enabled: true
    cpu_avg_pct: 85
    cpu_max_pct: 95
    mem_pct: 90
    lag_p95_ms: 50
```

## Resource collection

`report.collect_resources: true` samples the **load generator** host with
psutil (CPU%, RSS memory, scheduler lag) once per second into
`resources.loadgen`. `report.scrape` additionally pulls target-side
Prometheus metrics into `resources.targets` — use it to compare generator
health against target saturation in the same report.

## Reproducing the sample results

`docs/sample-results/` is generated, not hand-written:

```bash
python scripts/generate_sample_results.py
```

It starts a local mock OpenAI streaming server (12 chunks, ~30 ms apart),
runs a `smoke` profile for ~30 s, and writes CSV/JSON/HTML into
`docs/sample-results/`. Use it to sanity-check installs and to see the
full report shape (executive summary, bottleneck analysis, SLO rollup,
resource utilization, cost) without real provider credentials.

## Known limitations

- **Timestamp precision.** All timings are `time.monotonic()` wall clock;
  they include gevent scheduling delay (see above). Absolute values are
  accurate on an unloaded generator; relative comparisons across runs on
  the same generator remain meaningful.
- **Bedrock / gRPC / app_object** have coarser timing — TTFT/ITL are
  best-effort where the transport does not expose per-chunk timestamps.
- **Cost** is an estimate from `report.pricing` or provider-reported
  usage; it is not billing data.
- **Quality scores** are sampled (`quality.sample_rate`) and heuristic
  unless an LLM judge is configured — treat them as directional.
