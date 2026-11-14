# Resilience testing

`llm-perf` can validate graceful degradation at the dependency boundaries it
executes: direct LLM calls, RAG stages, agent LLM/tool calls, and HTTP/gRPC
application endpoints.

The feature combines four mechanisms:

- **Retries** — repeat transient failures with capped exponential backoff.
- **Circuit breakers** — stop calling a failing dependency after a threshold,
  then use a half-open probe to test recovery.
- **Fallbacks** — produce a bounded degraded result after failure or breaker
  rejection.
- **Fault injection** — inject deterministic failures for a configured time
  window so resilience behavior can be tested without waiting for a real outage.

All policies are disabled by default. Existing configurations keep their prior
behavior, including target-level `retries` and `retry_backoff_s`.

## Why it exists

A fast RAG or agent system can still fail badly when retrieval, embeddings,
tools, or model endpoints degrade. Resilience testing answers questions such as:

- Do transient failures recover after retries?
- Does a sustained outage open the breaker instead of increasing latency?
- Does the pipeline produce a controlled degraded response instead of an error?
- Does the breaker half-open and close after recovery?
- How much traffic is degraded, and does quality remain acceptable?

For framework-managed `pipeline` targets, `llm-perf` can directly measure these
behaviors. For an externally deployed `app_endpoint`, it validates observable
behavior only; it cannot prove the service's internal breaker state unless that
service exports telemetry.

## Configuration shape

Add `resilience` at target level for shared defaults, or on a pipeline stage for
a stage-specific override:

```yaml
targets:
  - name: rag_resilient
    layer: pipeline
    dataset: qa_pairs
    resilience:
      retry:
        attempts: 1
        backoff_s: 0.1
        max_backoff_s: 1
        jitter_s: 0
        retry_on: [timeout, connect, http_5xx]
      circuit_breaker:
        enabled: true
        failure_threshold: 3
        window_s: 30
        open_duration_s: 15
        half_open_calls: 1
        successes_to_close: 1

    pipeline:
      - stage: retrieve
        vector_store: { type: json, path: ../datasets/docs.jsonl }
        resilience:
          fallback: { type: empty_context, mark_degraded: true }

      - stage: generate
        llm:
          provider: http
          base_url: http://localhost:9001/chat
        resilience:
          fallback:
            type: alternate_llm
            llm:
              provider: http
              base_url: http://localhost:9002/chat
          fault_injection:
            enabled: true
            fault: http_503
            probability: 1.0
            after_s: 5
            duration_s: 20
```

A complete commented example lives at
[`configs/rag_resilience.yaml`](../configs/rag_resilience.yaml).

## Policies

### `retry`

| Field | Meaning |
|---|---|
| `attempts` | Number of retries after the first attempt. `0` disables retries. |
| `backoff_s` | Initial exponential backoff. |
| `max_backoff_s` | Backoff ceiling. |
| `jitter_s` | Random jitter added to each delay. |
| `retry_on` | Failure classes eligible for retry. |

Supported `retry_on` classes:

- `timeout`
- `connect`
- `http_4xx`
- `http_5xx`
- `llm_error`
- `empty_response`
- `exception`

For provider-backed LLM calls, `resilience.retry` maps onto the existing
provider retry implementation. If `resilience.retry.attempts` is not set, the
legacy target fields still work:

```yaml
targets:
  - name: openai_chat
    retries: 2
    retry_backoff_s: 0.5
```

### `circuit_breaker`

| Field | Meaning |
|---|---|
| `enabled` | Enable the closed/open/half-open state machine. |
| `failure_threshold` | Failures required inside the window to open. |
| `window_s` | Rolling failure-count window. |
| `open_duration_s` | Time before a half-open probe is allowed. |
| `half_open_calls` | Concurrent probe calls allowed while half-open. |
| `successes_to_close` | Successful probes required to close. |

Breaker state is keyed by `scenario / target / call site`. In a distributed
Locust run, each worker owns independent state.

### `fallback`

| Type | Typical use |
|---|---|
| `static_response` | Return configured `response` text. |
| `static_context` | Return configured `context` strings as retrieved context. |
| `empty_context` | Continue retrieval without context. |
| `passthrough` | Skip a degradable stage such as embedding/reranking. |
| `alternate_llm` | Call the LLM configuration under `fallback.llm`. |
| `alternate_tool` | Call the tool configuration under `fallback.tool`. |
| `alternate_endpoint` | Call another `app_endpoint` URL/path. |
| `none` | No fallback; propagate the failure. |

`mark_degraded: true` records fallback output as degraded traffic. Supported
fallbacks depend on the call site:

- `llm_api`: `static_response`, `alternate_llm`
- `retrieve`: `empty_context`, `static_context`, `passthrough`
- `embed`: `passthrough`, `empty_context`, `static_context`, `alternate_llm`
- `rerank`: `passthrough`, `static_context`, `alternate_llm`
- `generate` / agent LLM step: `static_response`, `alternate_llm`
- Agent tools: `static_response`, `alternate_tool`
- `app_endpoint`: `static_response`, `alternate_endpoint`

### `fault_injection`

Faults are disabled unless `enabled: true`. They apply only to calls executed by
`llm-perf`; they cannot inject failures inside an external service's private
dependencies.

| Field | Meaning |
|---|---|
| `fault` | `timeout`, `connect`, `http_500`, `http_503`, `exception`, `empty_response`, `llm_error`, `latency`, or `none`. |
| `probability` | Per-attempt injection probability, `0..1`. |
| `after_s` | Delay before the fault window starts. |
| `duration_s` | Optional fault window length; unset means inject indefinitely. |
| `latency_s` | Delay used by the `latency` fault. |
| `seed` | Optional deterministic RNG seed. |

A finite `after_s`/`duration_s` window creates a simple campaign in one run:
baseline traffic, failure traffic, then recovery traffic.

## How to run

Resilience uses the normal `run` command:

```bash
llm-perf validate -c configs/rag_resilience.yaml --skip-connectivity
llm-perf run -c configs/rag_resilience.yaml --headless
```

For deterministic local tests, point `base_url` values at controlled mock
services. Keep credentials in environment variables (`api_key_env`), not YAML.

## What reports show

The JSON report contains a `resilience` object keyed by call site, for example
`rag_resilience/rag_resilient/generate`:

- `retry_attempts`
- `retry_exhausted`
- `timeouts`
- `circuit_state`
- `circuit_opens`
- `circuit_half_opens`
- `circuit_closes`
- `circuit_rejections`
- `fallback_invocations`
- `fallback_successes`
- `fallback_failures`
- `degraded`
- `fault_injections`

Scenario rows also include aggregated fields such as
`retry_exhausted_total`, `circuit_open_total`, `circuit_rejections_total`,
`fallback_invocations_total`, `fallback_success_rate`, `degraded_rate`, and
`fault_injections_total`.

Resilience events are exported through OTel as `llm_perf.*` counters and the
`llm_perf.circuit.state` gauge (`0=closed`, `1=half_open`, `2=open`).

## Thresholds

Resilience SLOs can gate CI:

```yaml
thresholds:
  scenarios:
    rag_resilience:
      max_retry_exhausted: 10
      max_circuit_open_events: 2
      max_circuit_rejections: 100
      min_fallback_success_rate: 0.90
      max_degraded_rate: 1.0
      max_error_rate: 0.05
      min_availability: 0.95
      min_functional_availability: 0.90
      min_non_degraded_availability: 0.00  # fallback traffic is expected
      min_quality_score: 0.65
  on_fail: fail
```

`degraded_rate` is calculated from degraded dependency calls divided by scenario
requests. A request can contain multiple dependency calls, so use `1.0` when any
amount of fallback completion is acceptable.

## Boundaries

- `app_object` remains externally managed; it is not wrapped by the resilience
  engine in this MVP.
- `static_response` means the request can complete with degraded text; it is not
  a real provider result.
- A black-box `app_endpoint` run can measure fallback latency/output only for
  the configured local fallback endpoint. Internal resilience inside the remote
  service requires service-side instrumentation.
- Quality scoring still applies to fallback output, so degraded text can be
  compared against normal output.
