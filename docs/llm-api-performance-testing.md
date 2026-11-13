# LLM API performance testing with llm-perf

End-to-end guide to benchmarking **direct LLM API calls** (layer `llm_api`)
with this framework: OpenAI, Anthropic, Azure OpenAI, Bedrock, Google Gemini,
generic HTTP, gRPC — unary and streaming — plus Ollama and other
OpenAI-compatible local servers via `provider: openai` + `base_url`.

For RAG pipelines (retrieve → rerank → generate → agent) see
[rag-performance-testing.md](rag-performance-testing.md) instead; `llm_api`
is the single-call layer those pipelines build on.

## What gets measured

Every request through an `llm_api` target produces:

| Signal | Where it appears | What it tells you |
|---|---|---|
| End-to-end latency | `p50/p90/p95/p99` per scenario | User-visible response time |
| `TTFT` (time to first token) | `ttft_p95_ms` etc., only when `stream: true` | Prefill + queueing delay — the metric users feel first |
| `TPOT` (time per output token) | `tpot_p95_ms` | Decode speed; lower = faster streaming |
| `ITL` (inter-token latency) | `itl_avg_ms`, `itl_p95_ms` | Chunk cadence/jitter; spikes indicate provider or generator lag |
| `GEN_DURATION` | `gen_duration_p95_ms` | First→last token — total decode window |
| Throughput | `tokens_per_sec_avg`, `input_tokens_per_sec_avg`, `rps` | Decode rate and prompt-processing rate |
| `QUEUE_WAIT` | `queue_wait_p95_ms` | Task dispatch → request send; high values mean the **load generator** is saturated, not the provider |
| Token usage | `PROMPT_TOKENS`, `COMPLETION_TOKENS`, `TOKENS` | Context-window pressure and billing volume |
| Reliability | `error_rate`, `retries_total`, `timeouts` | Flaky-network/provider behavior under `retries`/`timeout_s` |
| `QUALITY_SCORE` / `QUALITY_PASS_RATE` | scenario stats + `quality_evaluated`/`quality_eligible`/`quality_judged`/`quality_judge_rate`, 95% CI | Heuristics + optional sampled LLM-judge on the answers |
| Cost | `cost_usd`, `cost_per_success_usd`, `cost_breakdown.generation` | Estimated spend when `report.pricing` is set |
| Latency composition | `composition` in JSON, stacked-share chart in HTML | Per-request median shares of e2e (e.g. `ttft` vs `generation` vs `queue_wait`) — never p95 subtraction |
| Traces | Jaeger (when OTel enabled) | Per-request span with model, token usage, finish reason, error |

The streaming split matters: `ttft` ≈ prefill+queue, `generation` ≈ decode.
A high TTFT share means prompt processing or provider queueing dominates; a
high generation share means decode-bound (long outputs, slow model).

## 1. Prepare your data

You need prompts. Three sources, usable alone or combined:

- **`prompts:`** — inline prompt pool on the target; `{var}` placeholders are
  filled from `vars:` pools or dataset row fields.
- **`dataset:`** — a dataset whose mapped `prompt` column supplies the query;
  extra columns are available to templates and quality heuristics.
- **Row-driven prompts** — a dataset column literally named `question`/`prompt`
  works without explicit `columns` mapping.

```jsonl
{"question": "Explain Kubernetes in one sentence.", "expected_answer": "A container orchestration platform.", "category": "short"}
{"question": "Write a haiku about {topic}.", "category": "creative"}
```

```yaml
targets:
  - name: openai_chat
    layer: llm_api
    vars:
      topic: ["kubernetes", "finops", "observability"]
```

Datasets load from local paths, `http(s)://`, `s3://`, `gs://` under the
`security.dataset_*` size/row/timeout limits, and are deterministically
sharded across distributed workers (`llm-perf dataset coverage -c <cfg> -w N`
verifies the partition before you spawn workers).

## 2. Configure a target

Full annotated example (based on `configs/openai_example.yaml`):

```yaml
observability:
  otel:
    enabled: true
    endpoint: http://localhost:4317
    service_name: llm-perf-openai
    sample_ratio: 1.0
    genai_semconv: true        # GenAI semantic conventions on spans

datasets:
  qa_pairs:
    source: ../datasets/qa_pairs.jsonl
    format: jsonl
    columns:
      prompt: question
      expected: expected_answer
    sample: { strategy: random, count: 50, seed: 7 }

targets:
  - name: openai_chat
    layer: llm_api
    provider: openai            # openai | anthropic | azure | bedrock | gemini | http | grpc
    base_url: https://api.openai.com/v1   # optional for openai/anthropic; required for azure/bedrock-style endpoints
    api_key_env: OPENAI_API_KEY           # env var NAME, never the key itself
    stream: true                 # SSE streaming -> TTFT/TPOT/ITL/gen-duration metrics
    stream_mode: full            # full = consume whole stream; first_token = stop after first chunk
    timeout_s: 120
    retries: 2                   # retry on timeout/connect/5xx
    retry_backoff_s: 0.5
    params:
      model: gpt-4o-mini
      max_tokens: 256
      temperature: 0.7
    dataset: qa_pairs
    prompts: ["Explain {topic} briefly."]   # optional inline pool
    # provider-specific blocks (all optional):
    # azure:   { api_version, deployment_field }
    # aws:     { region, auth, profile }
    # gemini:  { api_version }   # default v1beta; Vertex AI -> endpoint with {model}
    # grpc:    { proto_path, service, method, tls, ... }
    # request_template / response_mapping: for provider=http custom APIs

scenarios:
  - name: chat_load
    target: openai_chat
    weight: 100
    think_time: [0.5, 2.0]       # closed-loop pacing between requests
    stream: true                 # optional per-scenario override of target.stream
    quality:
      sample_rate: 0.2           # fraction sent to the LLM judge
      heuristics:
        non_empty: true
        min_chars: 10
        max_chars: 8000
        forbidden_patterns: ["I cannot assist", "as an AI language model"]
      judge:
        provider: openai
        model: gpt-4o
        rubric: "Score 1-5 on correctness, helpfulness, and conciseness"
        scale: [1, 5]

load:
  profile: load                  # smoke|baseline|load|stress|spike|soak|scalability|capacity|custom
  users: 10
  spawn_rate: 2
  duration: 5m
  ramp_up: 30s
  ramp_down: 30s

thresholds:
  global:
    max_error_rate: 0.02
    min_throughput_rps: 1
  scenarios:
    chat_load:
      p95_ms: 30000
      p99_ms: 60000
      ttft_p95_ms: 5000          # streaming SLO — NOT_EVALUATED on unary targets
      tpot_p95_ms: 100
      max_retries: 5
      min_quality_score: 0.6
      max_cost_usd: 5.0
  on_fail: fail
  on_not_evaluated: warn          # pass|warn|fail for rules with no data

report:
  formats: [csv, json, html]
  output_dir: results/openai
  collect_resources: true
  pricing: { input_per_1k: 0.00015, output_per_1k: 0.0006 }  # enables cost_usd
```

### Provider notes

- **`stream_mode: first_token`** — stops consuming after the first chunk;
  use it to measure TTFT/prefill cost without paying decode time.
- **Retries** apply to `timeout`, `connect`, and `5xx` failures; retried
  attempts that succeed are included in the response's token usage.
- **`provider: http`** targets arbitrary OpenAI-ish endpoints via
  `request_template` (Jinja2 body) + `response_mapping` — subject to the
  SSRF `security.url_allowlist`.
- **`provider: gemini`** — Google AI Studio (`generativelanguage.googleapis.com`,
  `x-goog-api-key` auth via `api_key_env: GEMINI_API_KEY`) **or Vertex AI**
  via the `gemini:` block:

  ```yaml
  gemini:
    vertex: true
    project: my-gcp-project      # optional — omit for express mode
    location: us-central1
    auth: bearer                 # api_key | bearer
  ```

  `vertex: true` switches the base URL to
  `https://<location>-aiplatform.googleapis.com` and the path to
  `/v1/projects/<p>/locations/<l>/publishers/google/models/{model}:…`
  (express mode drops project/location to `/v1/publishers/google/…`).
  `auth: bearer` sends `Authorization: Bearer <token>` — point
  `api_key_env` at an env var holding an OAuth2 access token (e.g.
  `gcloud auth print-access-token` or a service-account token from your
  secrets manager); `auth: api_key` keeps `x-goog-api-key` (express mode).
  `params` map onto `generationConfig` (`max_tokens`→`maxOutputTokens`,
  `top_p`→`topP`, …); `system` becomes `systemInstruction`. Streaming uses
  `streamGenerateContent?alt=sse` and yields TTFT/TPOT/ITL like other
  providers. See `configs/gemini_example.yaml` and
  `configs/gemini_vertex_example.yaml`.
- **Ollama / OpenAI-compatible local servers** — no dedicated provider
  needed: `provider: openai` + `base_url: http://localhost:11434/v1`
  (Ollama) or any LiteLLM/LM Studio/llama.cpp server exposing
  `/v1/chat/completions`. See `configs/ollama_example.yaml`.
- **`provider: grpc`** needs `grpc.proto_path`/`grpc.method` (or
  `proto-gen` to compile a bundled `.proto`).

## 3. How to run

```bash
# validate config (no network calls)
llm-perf validate -c configs/openai_example.yaml

# headless run
llm-perf run -c configs/openai_example.yaml --headless

# override shape
llm-perf run -c configs/openai_example.yaml --profile stress --headless \
    --users 50 --spawn-rate 5 --run-time 10m

# web UI (interactive)
llm-perf run -c configs/openai_example.yaml

# distributed
dataset coverage first:
llm-perf dataset coverage -c configs/openai_example.yaml -w 4
llm-perf run -c configs/openai_example.yaml --master 0.0.0.0 --headless   # terminal 1
llm-perf run -c configs/openai_example.yaml --worker <MASTER_HOST>        # terminals 2..n
```

API keys come from the env vars named by `api_key_env`. Never put keys in YAML.

### No-credentials dry run

`demo/all_layers.yaml` includes `llm_unary_load` and `llm_stream_load`
scenarios against `demo/mock_server.py` (OpenAI-compatible mock on `:8901`)
— see `demo/README.md`.

```bash
python demo/mock_server.py
LLM_PERF_CONFIG=demo/all_layers.yaml locust -f locustfile.py --headless -u 6 -r 6 -t 45s
```

## 4. View results

Reports land in `report.output_dir` as `llm-perf-<timestamp>.{csv,json,html}`.

**JSON** — the complete artifact. `llm_api`-relevant sections:

```jsonc
{
  "manifest": { "run_id": "…", "config": {"sha256": "…"},
                "targets": [{"name": "openai_chat", "provider": "openai",
                             "model": "gpt-4o-mini", "host": "api.openai.com"}], … },
  "scenarios": [{
    "scenario": "chat_load",
    "requests": 1240, "failures": 2, "error_rate": 0.0016,
    "p50_ms": 1450, "p95_ms": 6200, "p99_ms": 11800,
    "ttft_p95_ms": 820, "tpot_p95_ms": 18.4, "itl_p95_ms": 21.0,
    "gen_duration_p95_ms": 5100, "queue_wait_p95_ms": 35,
    "tokens_per_sec_avg": 61.2, "input_tokens_per_sec_avg": 5400,
    "total_tokens": 412000, "retries_total": 3, "timeouts": 0,
    "quality_score_avg": 0.84, "quality_evaluated": 1240,
    "quality_judged": 248, "quality_judge_rate": 0.2,
    "cost_usd": 0.62, "cost_per_success_usd": 0.0005
  }],
  "composition": {
    "chat_load": {"requests": 1240,
      "segments": {"ttft": {"median_share_pct": 13.2, "p95_ms": 820},
                   "generation": {"median_share_pct": 82.4, "p95_ms": 5100},
                   "queue_wait": {"median_share_pct": 0.3}},
      "unattributed": {"median_share_pct": 4.1}}
  },
  "cost_breakdown": {"chat_load": {"generation": 0.62}},
  "thresholds": {"all_passed": true, "failed": [], "not_evaluated": [], "rules": […]},
  "analysis": {"executive_summary": "…", "bottlenecks": ["decode_bound"]}
}
```

**How to read the streaming split:**

- `ttft` high share → prefill/queue-bound: shorter prompts, prompt caching,
  a faster-tier model, or provider-side queuing.
- `generation` (decode) high share → decode-bound: lower `max_tokens`,
  faster model tier, or streaming `first_token` mode if you only need TTFT.
- `queue_wait` high → the **generator** is the bottleneck (CPU/event-loop
  saturation), not the API — check `resources.loadgen.scheduler_lag`.

**CSV** — one row per scenario; **HTML** — verdict banner, per-scenario
table, per-request composition chart, bottleneck callouts, SLO rollup, cost
breakdown, resource utilization, baseline comparison.

**Live telemetry** (`docker compose up -d`): Grafana `:3000`, Jaeger `:16686`
(per-request span waterfall with GenAI attributes), Prometheus `:9090`.

## 5. Validate results

### Threshold states

Every rule reports `pass`, `fail`, or `not_evaluated`:

- A `ttft_p95_ms` rule on a **unary** (`stream: false`) target →
  `not_evaluated`, never a silent pass.
- `thresholds.on_not_evaluated: pass|warn|fail` controls CI treatment;
  `on_fail: fail` exits nonzero on any breach.

### Healthy signals for an `llm_api` run

| Check | Healthy | Investigate when |
|---|---|---|
| `error_rate` | < 1–2% | Retries/timeouts spiking → provider or network issue |
| `ttft_p95_ms` | stable across load | Grows with concurrency → provider queueing/prefill saturation |
| `tpot_p95_ms` / `itl_p95_ms` | flat vs load | Grows → decode slowdown or generator lag |
| `queue_wait_p95_ms` | ≈ think_time | ≫ think_time → load-generator saturation (`scheduler_lag` warning) |
| `tokens_per_sec` | scales with users | Flattens → provider rate-limit ceiling |
| `quality_judge_rate` | ≈ configured `sample_rate` | Much lower → judge errors or sampler bug |
| `cost_per_success_usd` | stable run-over-run | Drifts → pricing/model/prompt changes (see `manifest`) |

### Regression & provenance

Every report embeds a `manifest` (run UUID, git SHA + dirty flag, config &
dataset SHA256, provider/model/host, load shape, Python/hardware, pricing
snapshot). `llm-perf history` + `baseline mark` let you compare runs;
`comparison.provenance.drift` flags config/dataset/model changes so deltas
aren't mistaken for regressions.

## 6. End-to-end flow

1. **Prepare prompts** — inline pool, dataset `prompt` column, or `{var}`
   templating; add `expected_answer` if you want quality heuristics/judge.
2. **Pick the provider shape** — `stream: true` for TTFT/TPOT metrics;
   `stream_mode: first_token` for prefill-only cost; `retries`/`timeout_s`
   for flaky endpoints.
3. **Set SLOs** — latency percentiles + streaming metrics you actually
   configured; `on_not_evaluated: warn` while calibrating.
4. **Set pricing** — `report.pricing` turns token counts into `cost_usd`.
5. **Validate** — `llm-perf validate -c <cfg>` (catches schema + SSRF issues).
6. **Dry run** — `demo/all_layers.yaml` against the mock before spending
   real tokens.
7. **Run** — headless for CI, `--profile` for shape, distributed for scale.
8. **Read the report** — check `composition` shares first (where does e2e
   time go?), then thresholds verdict, then `cost_breakdown`.
9. **Gate CI** — `on_fail: fail` + baseline regression checks.
10. **Compare over time** — `history`/`baseline`; watch `provenance.drift`
    before trusting a delta.

## 7. Diagnostics

| Symptom | Likely cause | Where to look |
|---|---|---|
| TTFT p95 climbs with users, generation flat | Provider prefill/queue saturation | `composition.segments.ttft`, provider dashboards |
| TPOT/ITL p95 climbs, TTFT flat | Decode slowdown or event-loop lag | `itl_p95_ms`, `resources.loadgen.scheduler_lag` |
| `queue_wait` ≫ think_time | Generator CPU saturation | `resources.loadgen.cpu_percent`, lag probe |
| Error rate spikes at high concurrency | Rate limits (429) or timeouts | `retries_total`, `timeouts`, provider quotas |
| `quality_judge_rate` ≪ `sample_rate` | Judge call failures | `failure_tags: judge_error`, judge provider config |
| Cost higher than expected | Prompt bloat or judge spend | `cost_breakdown.judge`, `PROMPT_TOKENS` avg |
| `not_evaluated` rules | Metric not produced (e.g. TTFT on unary) | threshold rule vs `stream` setting |

## 8. Security notes

- `api_key_env` references env var **names** — keys never enter config,
  reports, or the manifest (secret-looking keys are redacted before hashing).
- `provider: http`/`app_endpoint` URLs go through the SSRF guard:
  `security.url_allowlist` + private-network blocking; redirects are
  re-validated.
- `security.trusted` / `security.python_execution` gate Jinja templates,
  Python tool imports, and `app_object` execution — the config audit warns
  on every run.
- Dataset fetches are capped by `security.dataset_max_bytes`/`max_rows`/
  `timeout_s`; responses are read with a capped, streaming read.
