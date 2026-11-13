# Performance testing black-box app endpoints (HTTP & gRPC) with llm-perf

This guide covers the **`app_endpoint` layer** — benchmarking an
application endpoint as a black box: you point the framework at an HTTP
URL or a gRPC `host:port`, and it measures **transport latency, throughput,
and error rate** under load. No provider SDK, no API-key semantics beyond a
Bearer header, no token accounting — the endpoint is treated as an opaque
service.

Use this layer when the thing under test is *your service* (a chat API, a
search endpoint, an internal microservice, a Triton/KServe inference
server) rather than a vendor LLM API. If you need TTFT/TPOT/tokens/quality
judging against an LLM protocol, use `layer: llm_api` instead (see
`llm-api-performance-testing.md`).

## 1. What gets measured

| Signal | Where it appears | Why it matters |
|---|---|---|
| End-to-end latency (p50/p90/p95/p99, avg) | `scenarios[].p50_ms…p99_ms`, `methods.POST`/`methods.<RPC>` | User-perceived response time of your endpoint |
| Throughput (`rps`) | `scenarios[].rps` | Requests/second the endpoint sustains |
| Error rate | `error_rate`, `failures` | HTTP ≥400, connection errors, gRPC errors, timeouts |
| Retries / timeouts | `retries_total`, `timeouts` | Resilience behavior under stress |
| Active users | `load.users` in report | Concurrency actually offered |
| Resource usage | `resources` (optional `collect_resources`) | Load-generator headroom — is the *harness* the bottleneck? |
| Threshold verdicts | `thresholds.results[]` | pass / fail / not_evaluated per rule |
| OTel traces/spans | Jaeger/OTel collector when enabled | Per-request span `app_endpoint` / `app_endpoint_grpc` |

What you do **not** get at this layer: token counts, TTFT/TPOT/ITL,
streaming split metrics, per-stage pipeline composition, or automatic
cost attribution — the endpoint's internals are invisible by design. You
*can* still attach quality heuristics and an LLM judge to the response
text (see §5).

## 2. Data prep

Two ways to feed requests:

- **`prompts:`** — inline list on the target; each entry may contain
  `{var}` placeholders filled from `vars:` pools (cartesian/random pick).
- **`dataset:`** — a named dataset from the top-level `datasets:` block.
  The picker takes the first scalar column named `prompt`, `question`,
  `input`, or `text` as the prompt; the **whole row** is also exposed to
  `request_template` as `{{ row.* }}` and scalar row fields as top-level
  template vars.

```yaml
datasets:
  chat_requests:
    source: ../datasets/chat_requests.jsonl   # {"prompt": "...", "user_id": "u1", ...}
    format: jsonl
    sample: { strategy: random, count: 200, seed: 7 }
```

Distributed runs shard datasets deterministically per worker —
`llm-perf dataset coverage -c <config> -w <N>` verifies collective
coverage (see `rag-performance-testing.md` §distributed).

## 3. Configure the target

### HTTP (black-box REST/gRPC-gateway/any HTTP service)

```yaml
targets:
  - name: chat_api
    layer: app_endpoint
    transport: http                # default
    base_url: https://api.example.com
    endpoint: /v1/chat             # appended to base_url
    method: POST                   # GET/PUT/DELETE also work
    timeout_s: 30
    retries: 1                     # retry on timeout/connect/5xx
    retry_backoff_s: 0.5
    headers:
      X-Tenant: demo
    api_key_env: APP_TOKEN         # -> Authorization: Bearer $APP_TOKEN
    request_template: |            # OPTIONAL — raw body template
      {"model": "demo", "messages": [{"role": "user", "content": "{{ prompt }}"}],
       "user": "{{ row.user_id }}"}
    dataset: chat_requests
    vars:
      topic: ["billing", "search", "onboarding"]
```

Request-body rules:

- **With `request_template`** — the Jinja2 template is rendered with
  `prompt`, `row`, and every scalar row field, then sent as the **raw
  request body** (bytes). Full control over method/path/headers/body —
  use it for non-JSON APIs too.
- **Without it** — a JSON body `{"prompt": <prompt>, **row}` is POSTed.

Response handling: the body is read with a hard byte cap
(`security.http_max_response_bytes`) and the first 20 000 chars become
the response text used for quality checks. Note: `response_mapping`
exists on the schema for `llm_api` HTTP providers but is **not applied**
at this layer — quality heuristics see the raw body text.

### gRPC (Triton, KServe, or any proto-defined service)

```yaml
targets:
  - name: triton
    layer: app_endpoint
    transport: grpc
    base_url: localhost:8001       # host:port (no scheme)
    timeout_s: 60
    grpc:
      proto_path: protos/inference.proto   # omit -> bundled Triton proto
      service: inference.GRPCInferenceService
      method: ModelInfer
      tls: false                    # true -> TLS channel credentials
      metadata: { "x-tenant": "demo" }
      request_fields:               # logical name -> proto field (best-effort)
        text: inputs.contents.string_contents
      response_fields:
        text: outputs
    params: { model: ensemble_model }
    prompts: ["Hello", "Ping {topic}"]
    vars: { topic: ["kubernetes", "finops"] }
```

gRPC workflow:

1. Put your `.proto` in the repo (e.g. `protos/my_service.proto`) or use
   the bundled `protos/inference.proto` (minimal Triton/KServe
   `ModelInfer`/`ModelStreamInfer`).
2. Pre-compile stubs once: `llm-perf proto-gen -p protos/my_service.proto`
   (uses `grpcio-tools`, a core dependency).
3. Set `grpc.proto_path` — stubs are compiled to `_generated/` next to
   the proto and imported dynamically.

Field mapping is **best-effort**: the adapter finds the `*Request`
message, fills fields named like `text`/`prompt`/`query`/`input` with
the prompt and `model`/`model_name` from `params.model`, then invokes
`service/method` (snake_case fallback). For services with unusual
schemas, verify one request manually before a load run — mismatches
surface as gRPC `INVALID_ARGUMENT` errors in `error_rate`.

## 4. Full example config

`demo/all_layers.yaml` contains a working `app_endpoint` target
(`chat_endpoint`) hitting the local mock's `/v1/chat/completions` via a
Jinja2 template — copy that shape for a real service:

```yaml
observability:
  otel: { enabled: true, endpoint: http://localhost:4317, service_name: llm-perf-app }

targets:
  - name: chat_endpoint
    layer: app_endpoint
    transport: http
    method: POST
    base_url: http://127.0.0.1:8901
    endpoint: /v1/chat/completions
    request_template: '{"model": "mock-model", "messages": [{"role": "user", "content": "{{ prompt }}"}], "max_tokens": 32}'
    response_mapping: { text: "choices.0.message.content" }   # not applied at this layer
    prompts: ["Ping the endpoint", "Say hi to the API"]

scenarios:
  - name: endpoint_load
    target: chat_endpoint
    weight: 100
    think_time: [0.1, 0.3]
    quality:
      sample_rate: 0.5
      heuristics: { non_empty: true, min_chars: 5 }

load:
  profile: smoke
  users: 5
  spawn_rate: 5
  duration: 2m

thresholds:
  on_fail: fail                  # non-zero exit on SLO breach (CI gating)
  scenarios:
    endpoint_load:
      p95_ms: 2000
      max_error_rate: 0.02

report:
  formats: [csv, json, html]
  output_dir: results/app-endpoint
  collect_resources: true
```

## 5. Quality on black-box responses

`scenario.quality` works at this layer because heuristics/judge operate
on the response **text**:

- `heuristics` — `non_empty`, `min_chars`, `max_chars`, `json_valid`,
  `contains` (substring list), `required_patterns`/`forbidden_patterns`
  (regex), `exact_match_field` (dataset column to compare the whole
  response against).
- `judge` — an LLM-as-judge call scored against `rubric`; counts toward
  `cost_breakdown.judge` if `report.pricing` is set.

Caveat: without `response_mapping`, JSON APIs feed the **whole body**
(e.g. `{"choices":[...]}`) to the checks — `non_empty`/`min_chars`/
`json_valid`/`contains` are fine, but `exact_match_field` compares the
entire raw body. For precise field extraction, prefer `layer: llm_api`
+ `provider: http` (which *does* apply `response_mapping`).

## 6. How to run

```bash
# validate config (no traffic)
llm-perf validate -c configs/app_endpoint_example.yaml

# headless run
llm-perf run -c configs/app_endpoint_example.yaml --headless

# overrides without editing YAML
llm-perf run -c configs/app_endpoint_example.yaml --headless \
  --users 20 --spawn-rate 5 --duration 5m --profile stress

# distributed: master + N workers (datasets shard automatically)
llm-perf run -c configs/app_endpoint_example.yaml --master=0.0.0.0 --workers 4
llm-perf run -c configs/app_endpoint_example.yaml --worker --master <master-host>

# no-credentials smoke: the demo mock serves /v1/chat/completions
python demo/mock_server.py &
llm-perf run -c demo/all_layers.yaml --headless --scenarios endpoint_load
```

`--layers app_endpoint` filters a mixed config to just endpoint targets.

## 7. View results

Reports land in `report.output_dir` as `llm-perf-<ts>.{csv,json,html}`.

**JSON** — per-scenario block:

```jsonc
{
  "scenarios": [{
    "scenario": "endpoint_load",
    "requests": 1830, "failures": 3, "error_rate": 0.0016,
    "rps": 15.2,
    "avg_ms": 41.2, "p50_ms": 38.0, "p90_ms": 61.5,
    "p95_ms": 88.4, "p99_ms": 140.1,
    "retries_total": 2, "timeouts": 1,
    "methods": { "POST": { "requests": 1830, "p95": 90, ... } },
    "quality_score_avg": 0.97, "quality_pass_rate": 1.0
  }],
  "stages": [],                    // no stage metrics at this layer
  "composition": { "endpoint_load": { "unattributed": {...} } },
  "thresholds": { "results": [ {"name": "scenario:endpoint_load.p95_ms",
                                 "passed": true, "actual": 88.4, "threshold": 2000} ] }
}
```

**HTML** — request-rate/error-rate charts, latency percentile panel,
threshold table, quality section.

**Grafana/Jaeger** — with `observability.otel.enabled`, each request
emits an `app_endpoint` (or `app_endpoint_grpc`) span plus
`llm_perf.request_duration`/`request_total` metrics for the provisioned
dashboards (`deployments/grafana`, `deployments/otel`).

## 8. Validate results

Healthy-signal checklist:

| Check | Healthy | Investigate when |
|---|---|---|
| `error_rate` | ≈ 0 | Any sustained >0 — read `methods.*`/span `error` attrs; check `thresholds` for which rule tripped |
| `p95_ms` vs threshold | comfortably under | Close to/over → endpoint saturated or network-bound; compare `resources` (is the *generator* maxed?) |
| `rps` plateau | scales with users until server limits | Flat rps + rising latency = server-side queueing, not client limit |
| `timeouts`/`retries_total` | 0 | Retries inflate e2e — check `timeout_s` vs p99 |
| `quality_score_avg` | ≥ your `min_quality_score` | Dropped under load → endpoint degrading semantically, not just slower |
| `not_evaluated` rules | none unexpected | e.g. a `ttft_p95_ms` rule here is meaningless — this layer has no streaming split |

Regression comparison: `report.baseline` (`previous`/`designated`/`none`)
+ `report.regression` thresholds compare the run against history; the
`manifest` block (git commit, config + dataset hashes) tells you whether
two runs are comparable.

## 9. End-to-end flow

1. **Identify the endpoint** — HTTP URL+path or gRPC `host:port`+proto.
2. **Prepare inputs** — `prompts:` pool or a `dataset:`; add `vars:` for
   template variety; keep PII out of committed datasets.
3. **Write the config** — copy §4; set `request_template` if the body
   isn't `{prompt, ...row}` JSON.
4. **gRPC only** — `llm-perf proto-gen -p protos/x.proto`, set
   `grpc.proto_path/service/method`, smoke-test one call.
5. **Validate** — `llm-perf validate -c …` (catches schema/URL issues).
6. **Smoke** — `--users 2 -t 30s` against a non-prod instance.
7. **Full run** — chosen `--profile` or explicit load; keep
   `collect_resources: true` to rule out generator bottlenecks.
8. **Read the report** — latency percentiles, error rate, thresholds,
   quality; open HTML for charts.
9. **Compare** — baseline vs candidate via `report.compare_to` or the
   history index; check `manifest` provenance drift.
10. **Gate CI** — `thresholds.on_fail: fail` → non-zero exit on breach.

## 10. Diagnostics

| Symptom | Likely cause | Where to look |
|---|---|---|
| High `error_rate`, `HTTP 5xx` | Endpoint overloaded or buggy under concurrency | Server logs; lower `--users` to find knee |
| High `error_rate`, timeouts only | `timeout_s` too tight for tail latency | Raise `timeout_s`; compare p99 vs timeout |
| `rps` flat, latency climbing | Server saturation (connection pool/CPU) | Server metrics; `resources` for generator headroom |
| All requests fail instantly | Wrong `base_url`/`endpoint`/`method`, or TLS mismatch (gRPC `tls:` flag) | One manual `curl`/`grpcurl` call |
| gRPC `INVALID_ARGUMENT`/`UNIMPLEMENTED` | proto/method/field mapping wrong | `grpc.method`, `service`, `request_fields`; check stub compiled |
| Quality fails but latency fine | Response shape changed / template stale | Raw body in report; heuristics see unmodified text |
| Latency fine in smoke, bad at scale | Connection reuse/keep-alive, DNS, LB | Compare per-phase metrics; distributed run |

## 11. Security notes

- Every request URL is validated by `security.check_url` — including
  each redirect hop (`CheckedHTTPAdapter`); `security.url_allowlist`
  restricts hosts, `security.allow_private_networks: false` blocks
  RFC1918/loopback/metadata IPs (SSRF guard for untrusted configs).
- `api_key_env` reads the credential from an **environment variable** —
  never put secrets in YAML; only the env var *name* is configured.
- Response bodies are capped (`security.http_max_response_bytes`) and
  truncated to 20k chars — hostile endpoints can't exhaust memory.
- `request_template` is Jinja2 over prompt/row data — it can produce an
  arbitrary body, so `audit_config` flags it in the untrusted-config
  warning. Only run configs you trust.
- gRPC `metadata` values are static config — don't place tokens there;
  there is currently no env-var expansion for gRPC metadata.
- Datasets fetched from `http(s)`/`s3`/`gs` sources are capped and go
  through the same URL policy.
