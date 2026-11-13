# Performance testing RAG pipelines with llm-perf

This guide walks through benchmarking a Retrieval-Augmented Generation (RAG)
system end-to-end: configuring the declarative pipeline, running load tests,
reading the reports, validating quality and latency SLOs, and wiring the whole
flow into CI. Everything below uses real config keys from this repository —
the working example is `configs/rag_example.yaml` and a no-credentials demo is
`demo/all_layers.yaml`.

## What the framework measures for RAG

A `layer: pipeline` target executes stages in order — `retrieve` → `rerank` →
`generate` (or `agent`) — and records **per-stage latency** plus end-to-end
metrics for every request:

| Signal | Where it appears | What it tells you |
|---|---|---|
| Stage latency (`retrieve`, `rerank`, `generate`, `agent_step`, `agent_tool`) | `stages[]` in JSON, stage table in HTML, `stage.<name>.p95_ms` threshold rules | Which stage is the bottleneck — embedding/search vs LLM decode vs tool calls |
| End-to-end latency | `p50/p90/p95/p99` per scenario | User-visible response time |
| `CONTEXT_RECALL` | scenario stats + `min_context_recall` threshold | Fraction of `expected_context` found in retrieved docs — **retrieval quality** |
| `RECALL_AT_K` / `PRECISION_AT_K` / `MRR` / `NDCG` | scenario stats (`recall_at_k`, `precision_at_k`, `mrr`, `ndcg`) + `min_recall_at_k`/`min_precision_at_k`/`min_mrr`/`min_ndcg` threshold rules | Standard IR metrics over the retrieved ranking. Require relevance judgments on the row: `relevant_ids` (doc ids) or `relevant_texts` (passages); falls back to `expected_context` part-coverage. No judgments → metrics absent (not evaluated), never a misleading 0 |
| `QUALITY_SCORE` | scenario stats + evaluated/eligible/judged counts, eval & judge rates, 95% CI + `min_quality_score` threshold | Heuristics + optional LLM-judge score — **answer quality**. `quality_evaluated`/`quality_eligible` (heuristics run on every successful response) and `quality_judged`/`quality_judge_rate` (realized LLM-judge sampling) show exactly how much was measured |
| Token metrics | `PROMPT_TOKENS`, `COMPLETION_TOKENS`, `TOKENS_PER_SEC`, `INPUT_TOKENS_PER_SEC` | Context-window pressure and decode throughput |
| Cost | `cost_usd`, `cost_per_success_usd`, `cost_breakdown` (when `report.pricing` set) | $ per run / per successful answer, split by call site: `generation`, `reranking`, `embedding`, `agent`, `judge`, `retry`, `unattributed` |
| Latency composition | `composition` in JSON, stacked-share chart in HTML | Per-request stage shares of e2e (median across requests). **Percentiles are never subtracted** — shares are computed inside each request first, then aggregated |
| Agent stats (agent stage only) | `AGENT_STEPS`, `AGENT_TOOL_CALLS`, `AGENT_SUCCESS`, `AGENT_LOOPS` | Tool-loop behavior, iterations, loop detection |
| Traces | Jaeger (when OTel enabled) | Per-request span waterfall: retrieve → rerank → generate |

The key RAG insight: **retrieval quality and generation quality are measured
separately**. `context_recall`/IR metrics isolate whether the right documents
were found; `quality_score`/judge isolates whether the answer used them well.

## 1. Prepare your data

You need two things: a **question set** and a **document store**.

### Question set (`datasets/qa_pairs.jsonl`)

```json
{"question": "What is Kubernetes?", "expected_answer": "An open-source container orchestration platform.", "expected_context": "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.", "category": "short"}
```

- `question` — the user query sent through the pipeline
- `expected_answer` — optional ground truth for heuristics/judge
- `expected_context` — **the passage that *should* be retrieved**; drives the
  `CONTEXT_RECALL` metric and the `expected_context` fallback for IR metrics.
  Include it whenever you can.
- `relevant_ids` — **preferred for IR metrics**: list of document ids (matching
  `id`/`_id`/`doc_id`/`uuid` fields in your store, or `idx:N` positional ids)
  that are relevant to this question. Enables exact Recall@K/Precision@K/MRR/NDCG.
- `relevant_texts` — alternative: list of relevant passages; a retrieved doc
  counts as relevant on normalized text containment.
- Any extra columns (`category`, `difficulty`, …) are available to Jinja
  templates and `vars` pools.

Column mapping in config renames source fields to logical names:

```yaml
datasets:
  qa_pairs:
    source: ../datasets/qa_pairs.jsonl
    format: jsonl                       # yaml | csv | jsonl (auto-detected)
    columns:
      prompt: question                 # logical name <- source field
      expected: expected_answer
      context: expected_context
    sample: { strategy: random, count: 50, seed: 11 }   # first|random|weighted
```

`source` accepts local paths, `http(s)://`, `s3://`, `gs://` — all subject to
the `security.dataset_*` size/row/timeout limits.

### Document store

Three `vector_store.type` options:

| Type | Config | Notes |
|---|---|---|
| `json` | `{type: json, path: docs.jsonl}` | Local JSON/JSONL docs; **keyword-overlap scoring, no embeddings** — fine for framework testing, not representative of a real vector DB |
| `http` | `{type: http, url: "...", api_key_env: X, headers: {...}, params: {...}}` | POSTs `{query, top_k}`; normalizes `results`/`documents`/`chunks`/list responses. Point at your real retrieval service (Pinecone/Qdrant/Weaviate proxy, Elasticsearch, etc.) |
| `chroma` | `{type: chroma, path: ./chroma_db, collection: llm_perf}` | Local Chroma via `pip install chromadb` |

For the `json` type each doc needs a `text` (or `content`) field; `score` and
extra metadata are passed through. For `http`, docs accept `text`, `content`,
`page_content`, or `score`.

## 2. Configure the pipeline

Full annotated config (`configs/rag_example.yaml`):

```yaml
observability:
  otel:
    enabled: true
    endpoint: http://localhost:4317
    service_name: llm-perf-rag

datasets:
  qa_pairs:
    source: ../datasets/qa_pairs.jsonl
    format: jsonl
    columns:
      prompt: question
      expected: expected_answer
      context: expected_context
    sample: { strategy: random, count: 50, seed: 11 }

targets:
  - name: rag_pipeline
    layer: pipeline
    dataset: qa_pairs                    # rows drive prompts + expected fields
    pipeline:
      - stage: retrieve
        vector_store:
          type: json                     # or http / chroma
          path: ../datasets/docs.jsonl
        top_k: 3                         # docs passed to rerank/generate

      - stage: rerank                    # OPTIONAL — omit to skip
        method: llm                      # llm | cross_encoder
        llm: { provider: openai, model: gpt-4o-mini, api_key_env: OPENAI_API_KEY }

      - stage: generate
        template: ../prompts/rag_answer.j2
        llm:
          provider: openai
          model: gpt-4o-mini
          api_key_env: OPENAI_API_KEY
          params: { max_tokens: 512, temperature: 0.2 }

scenarios:
  - name: rag_load
    target: rag_pipeline
    weight: 100
    think_time: [1.0, 3.0]               # seconds between requests (closed-loop)
    quality:
      sample_rate: 0.3                   # fraction of requests quality-scored
      heuristics:
        non_empty: true
        min_chars: 10
        forbidden_patterns: ["I don't have enough information"]
      rag: {}                            # enables CONTEXT_RECALL vs expected_context
      judge:                             # OPTIONAL LLM-as-judge (costs tokens)
        provider: openai
        model: gpt-4o
        rubric: "Score 1-5 on faithfulness to the context and correctness"
        scale: [1, 5]

load:
  profile: load                         # smoke|baseline|load|stress|spike|soak|scalability|capacity|custom
  users: 5
  spawn_rate: 1
  duration: 5m
  ramp_up: 30s
  ramp_down: 30s

thresholds:
  on_fail: fail                         # fail -> nonzero CI exit
  on_not_evaluated: warn                # pass|warn|fail for rules with no data
  scenarios:
    rag_load:
      p95_ms: 60000                     # end-to-end SLO
      max_error_rate: 0.05
      min_quality_score: 0.6            # judge/heuristic composite >= 0.6
      min_context_recall: 0.5           # retrieval recall floor
      min_recall_at_k: 0.6              # IR metrics (need relevant_ids/texts/expected_context)
      min_precision_at_k: 0.3
      min_mrr: 0.5
      min_ndcg: 0.5
      stages:                           # per-stage SLOs
        retrieve: { p95_ms: 2000 }
        generate: { p95_ms: 40000 }

report:
  formats: [csv, json, html]
  output_dir: results/rag
  collect_resources: true
  pricing: { input_per_1k: 0.00015, output_per_1k: 0.0006 }   # enables cost_usd
```

### Stage details

- **`retrieve`** — `vector_store` + `top_k`. Retrieved docs land in
  `ctx.context_texts` and `ctx.docs` for downstream stages. Emits a
  `retrieve` stage-timing metric per request.
- **`rerank`** — `method: llm` scores each doc 0–10 with the configured LLM
  (pointwise, one call per doc); `cross_encoder` uses a local model. Reorders
  `ctx.docs`; the reranked texts become the generation context.
- **`generate`** — renders `template` (Jinja2) with `question`, `context`
  (joined retrieved texts), `row` (dataset row), and every row field as a
  variable, then calls `llm`. Dataset columns literally named `question`,
  `context`, or `row` are reserved — they won't override the pipeline values.
- **`agent`** — tool-calling loop instead of generate; see
  `configs/agent_example.yaml`.

### Prompt template (`prompts/rag_answer.j2`)

```jinja
Answer the user's question using ONLY the context below. If the context does not contain the answer, say so.

Context:
{{ context }}

Question: {{ question }}

Answer:
```

`{{ context }}` = retrieved docs joined by blank lines; `{{ question }}` =
the dataset row's prompt field; any other `{{ var }}` resolves from row fields
then dataset/target `vars` pools.

## 3. Run the test

```bash
# install (once)
pip install -e .
# optional, only for chroma vector stores (not a package extra):
pip install chromadb

# validate config before spending tokens
llm-perf validate -c configs/rag_example.yaml

# quick smoke (5 users, 2 minutes)
llm-perf run -c configs/rag_example.yaml --headless \
  --users 5 --spawn-rate 1 --duration 2m

# override the profile shape entirely
llm-perf run -c configs/rag_example.yaml --profile stress --headless

# web UI mode (interactive, no --headless)
llm-perf run -c configs/rag_example.yaml
```

API keys come from environment variables named by `api_key_env`
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, AWS creds for
Bedrock). Never put keys in the YAML.

### Distributed runs

```bash
# terminal 1 — master (aggregates + writes reports)
llm-perf run -c configs/rag_example.yaml --master 0.0.0.0 --headless

# terminals 2..n — workers
llm-perf run -c configs/rag_example.yaml --worker <MASTER_HOST> --headless
```

Datasets are deterministically sharded across workers; reports/thresholds
fire only on the master. Before spawning workers, verify coverage:

```bash
llm-perf dataset coverage -c configs/rag_example.yaml -w 4
# qa_pairs: 50 rows / 4 workers -> per-worker=[13, 13, 12, 12] covered=50 OK
```

This proves the `i % workers` partition covers every row exactly once (no
gaps, no duplicates). Each worker's manifest records the dataset SHA256, so
runs against different dataset versions are detectable in `comparison`.

### No-credentials demo

`demo/rag_dryrun.yaml` runs a focused retrieve → rerank → generate pipeline
against `demo/mock_server.py` (OpenAI-compatible mock on `:8901`) with
`relevant_ids` judgments, sampled judge, and IR threshold rules — see
`demo/README.md` for the full reproduce flow.

## 4. View results

Reports land in `report.output_dir` as `llm-perf-<timestamp>.{csv,json,html}`.

**JSON** — the complete artifact. RAG-relevant sections:

```jsonc
{
  "manifest": { "run_id": "…", "config": {"sha256": "…"},
                "datasets": {"qa_pairs": {"sha256": "…"}}, … },
  "scenarios": [{
    "scenario": "rag_load",
    "requests": 412, "failures": 3, "error_rate": 0.007,
    "p50_ms": 4200, "p95_ms": 38500, "p99_ms": 61000,
    "ttft_p95_ms": null,          // null for non-streaming generate
    "total_tokens": 98400, "tokens_per_sec_avg": 412.5,
    "cost_usd": 0.1482, "cost_per_success_usd": 0.00036,
    "quality_evaluated": 412, "quality_eligible": 412, "quality_judged": 82,
    "quality_sample_rate": 1.0, "quality_judge_rate": 0.2,
    "quality": {"score_avg": 0.81, "score_ci95": 0.03, …},
    "recall_at_k": 0.72, "precision_at_k": 0.36, "mrr": 0.68, "ndcg": 0.61
  }],
  "stages": [
    {"scenario": "rag_load", "stage": "retrieve", "count": 412,
     "avg_ms": 1.8, "p95_ms": 4.2, "max_ms": 31.0},
    {"scenario": "rag_load", "stage": "generate", "count": 412,
     "avg_ms": 3900.0, "p95_ms": 37000.0, "max_ms": 58000.0}
  ],
  "composition": {
    "rag_load": {"requests": 412,
      "segments": {"retrieve": {"p50_ms": 1.2, "p95_ms": 4.2, "median_share_pct": 0.05},
                   "generate": {"p50_ms": 3900.0, "p95_ms": 37000.0, "median_share_pct": 0.56}},
      "unattributed": {"p50_ms": 800.0, "p95_ms": 15000.0, "median_share_pct": 0.39}}
  },
  "cost_breakdown": {
    "rag_load": {"generation": 0.12, "reranking": 0.02, "judge": 0.008, "unattributed": 0.0004}
  },
  "analysis": {
    "executive_summary": "…", "slo": {…}, "bottlenecks": ["decode_bound"]
  },
  "thresholds": {"all_passed": true, "failed": [], "not_evaluated": [], "rules": […]},
  "comparison": {"provenance": {"drift": [], "comparable": true}, …},
  "resources": {"loadgen": {"cpu_percent": …, "mem_pct": …}}
}
```

**CSV** — one row per scenario; filter/pivot in a spreadsheet.

**HTML** — verdict banner, per-scenario table, latency-composition chart
(**per-request median shares** of e2e — retrieval/generation/queue segments,
never p95 subtraction), stage breakdown, bottleneck callouts, SLO rollup,
quality table (evaluated counts, sample rate, 95% CI, IR metrics), cost +
cost-by-call-site breakdown, resources, baseline comparison. Open it in a
browser.

**Live telemetry** (when `docker compose up -d` is running):

- Grafana `http://localhost:3000` — request rate, latency heatmaps, token
  throughput, stage durations, quality scores
- Jaeger `http://localhost:16686` — per-request traces showing the
  retrieve → rerank → generate span waterfall and LLM call attributes
  (model, token usage, finish reason)
- Prometheus `http://localhost:9090` — raw metrics incl. custom stage stats

## 5. Validate results

### Threshold states

Every rule reports `pass`, `fail`, or `not_evaluated` (missing data — e.g. a
TTFT rule on a non-streaming target). `thresholds.on_not_evaluated` controls
whether unevaluated rules pass (`pass`, default), warn (`warn`), or fail CI
(`fail`). `on_fail: fail` makes any breached rule exit nonzero.

### What "good" looks like for RAG

| Check | Healthy signal |
|---|---|
| `retrieve` p95 | Tens of ms for `json` store; tens–hundreds of ms for real vector DBs; dominated by embedding latency for `chroma` |
| `generate` p95 | ≈ end-to-end p95 minus retrieve — LLM decode dominates |
| `CONTEXT_RECALL` | ≥ your `min_context_recall` floor; < 0.3 usually means retrieval (embeddings/top_k/store) is broken, not the LLM |
| `QUALITY_SCORE` | ≥ `min_quality_score`; low recall + high quality = retrieval problem; high recall + low quality = prompt/model problem |
| Bottleneck label | `decode_bound` = LLM-bound (scale concurrency carefully, watch rate limits); `retrieval_bound`/queueing = store-bound |
| `error_rate` | < 1%; spikes at high concurrency often mean provider 429s — check `retries`/`timeouts` |
| Cost | `cost_per_success_usd` — rerank stage multiplies LLM calls (one per doc); watch it at scale |
| Resources | `resources.loadgen` CPU/mem + scheduler lag — a saturated generator invalidates latency numbers |

### Baselines & regression

```bash
# after a known-good run
llm-perf baseline-mark -c configs/rag_example.yaml

# subsequent runs auto-compare; report.comparison + report.regression appear
llm-perf run -c configs/rag_example.yaml --headless
llm-perf baseline-list -c configs/rag_example.yaml
```

`comparison.provenance.drift` flags when the config hash, dataset hash, git
commit, or framework version differs from the baseline — treat deltas across
drifted runs as indicative only. `regression.on_fail: fail` turns detected
regressions into a nonzero exit.

## 6. End-to-end flow (checklist)

1. **Prepare data** — question set with `expected_context`; doc store sized to
   your corpus; pick `vector_store.type` (`http` for a real service).
2. **Write config** — start from `configs/rag_example.yaml`; set datasets,
   vector store, stage LLMs (`api_key_env`, never literal keys), quality
   sampling, thresholds incl. per-stage rules, pricing.
3. **Validate** — `llm-perf validate -c …` (checks schema + scenario/target/
   dataset references). Review the security audit warnings it prints.
4. **Smoke** — 1–2 users, 1–2 min, generous thresholds. Fix config errors
   here, not at scale.
5. **Load test** — chosen profile (`load`/`stress`/`soak`) or explicit
   `users`/`spawn_rate`/`duration`; add `ramp_up`/`ramp_down` for realism.
6. **Observe** — Grafana dashboards + Jaeger traces while it runs; check the
   load generator isn't the bottleneck (`resources.loadgen`).
7. **Read reports** — JSON for numbers, HTML for stakeholders; check
   `stages[]`, `CONTEXT_RECALL`, `QUALITY_SCORE`, thresholds verdict,
   bottlenecks, cost, manifest provenance.
8. **Validate quality** — is retrieval finding the right docs (recall)? Is
   the answer faithful (judge score + forbidden patterns)? Did any stage SLO
   break?
9. **Baseline & gate** — `baseline-mark` a good run; wire
   `on_fail: fail` + `regression.on_fail: fail` into CI so merges can't
   silently degrade RAG quality or latency.
10. **Iterate** — adjust `top_k`, rerank on/off, prompt template, model, or
    concurrency; rerun and compare via `comparison`/`provenance`.

## 7. Interpreting common RAG outcomes

| Symptom | Likely cause | Where to look |
|---|---|---|
| High e2e p95, `generate` p95 ≈ e2e | LLM decode-bound | `stages[]`, bottleneck `decode_bound`; reduce `max_tokens`, pick faster model, or cap concurrency |
| `retrieve` p95 spikes under load | Vector store saturation | `http` store metrics/Jaeger retrieve span; lower `top_k`, add store replicas, or shard index |
| Low `CONTEXT_RECALL`, answers say "no information" | Retrieval miss | `expected_context` mapping, embeddings/index freshness, `top_k` too small |
| Recall fine, judge score low | Prompt/model issue | `rag_answer.j2` wording, `temperature`, judge rubric calibration |
| `QUALITY_SCORE` null | `sample_rate: 0` or quality block missing | scenario `quality:` config |
| Huge `cost_usd` | Rerank fan-out + long contexts | rerank stage = N extra LLM calls/request; trim `top_k`, cheaper rerank model |
| All requests fail with URL errors | Security policy | `security.url_allowlist` / `allow_private_networks` blocking the store endpoint |
| `not_evaluated` rules | Metric never produced | e.g. `ttft_p95_ms` on non-streaming; set `on_not_evaluated` policy deliberately |

## 8. Security notes for RAG configs

- Vector-store `http` calls go through the checked HTTP session — the URL
  must pass `security.url_allowlist` and redirect hops are re-validated.
- `security.python_execution` gates `app_object` imports; agent `function`
  tools need it too.
- Dataset size/row/timeout caps apply to every source including `s3://`/`gs://`.
- Secrets only via `api_key_env` env vars; the manifest redacts secret-looking
  keys before hashing and reports URL hosts without userinfo.
- Startup prints an audit of powerful features (HTTP tools, Jinja templates,
  Python imports) — only run configs you trust; set `security.trusted: true`
  to acknowledge.
