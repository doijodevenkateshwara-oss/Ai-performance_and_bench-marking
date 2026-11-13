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
| Stage latency (`embed`, `retrieve`, `rerank`, `generate`, `agent_step`, `agent_tool`) | `stages[]` in JSON, stage table in HTML, `stage.<name>.p95_ms` threshold rules | Which stage is the bottleneck — query embedding/search vs LLM decode vs tool calls |
| End-to-end latency | `p50/p90/p95/p99` per scenario | User-visible response time |
| `CONTEXT_RECALL` | scenario stats + `min_context_recall` threshold | Fraction of `expected_context` found in retrieved docs — **retrieval quality** |
| `RECALL_AT_K` / `PRECISION_AT_K` / `MRR` / `NDCG` | scenario stats (`recall_at_k`, `precision_at_k`, `mrr`, `ndcg`) + `min_recall_at_k`/`min_precision_at_k`/`min_mrr`/`min_ndcg` threshold rules | Standard IR metrics over the retrieved ranking. Require relevance judgments on the row: `relevant_ids` (doc ids) or `relevant_texts` (passages); falls back to `expected_context` part-coverage. No judgments → metrics absent (not evaluated), never a misleading 0 |
| `QUALITY_SCORE` | scenario stats + evaluated/eligible/judged counts, eval & judge rates, 95% CI + `min_quality_score` threshold | Heuristics + optional LLM-judge score — **answer quality**. `quality_evaluated`/`quality_eligible` (heuristics run on every successful response) and `quality_judged`/`quality_judge_rate` (realized LLM-judge sampling) show exactly how much was measured |
| Token metrics | `PROMPT_TOKENS`, `COMPLETION_TOKENS`, `TOKENS_PER_SEC`, `INPUT_TOKENS_PER_SEC` | Context-window pressure and decode throughput |
| Cost | `cost_usd`, `cost_per_success_usd`, `cost_breakdown` (when `report.pricing` set) | $ per run / per successful answer, split by call site: `generation`, `reranking`, `embedding`, `agent`, `judge`, `retry`, `unattributed` |
| Latency composition | `composition` in JSON, stacked-share chart in HTML | Per-request stage shares of e2e (median across requests). **Percentiles are never subtracted** — shares are computed inside each request first, then aggregated |
| Agent stats (agent stage only) | `AGENT_STEPS`, `AGENT_TOOL_CALLS`, `AGENT_TOOL_ERRORS`, `AGENT_LOOPS`, `AGENT_SUCCESS`, per-tool stats | Tool-loop behavior, iterations, loop detection, task success |
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
| `http` | `{type: http, url: "...", api_key_env: X, headers: {...}, params: {...}}` | POSTs `{query, top_k}` — plus `query_vector` when an `embed` stage ran first; normalizes `results`/`documents`/`chunks`/`matches`/list responses (incl. Pinecone-style `{id, score, metadata}`). Point at your real retrieval service |
| `chroma` | `{type: chroma, path: ./chroma_db, collection: llm_perf}` | Local Chroma vector store — **only install chromadb if you actually benchmark a Chroma-backed setup** (see note below) |

For the `json` type each doc needs a `text` (or `content`) field; `score` and
extra metadata are passed through. For `http`, docs accept `text`, `content`,
`page_content`, or `score`.

> **Choosing a store — what you're actually benchmarking.** The `retrieve`
> stage measures *your retrieval dependency's per-request latency* as part of
> the pipeline — it is not a profiler of the store itself. Pick the backend
> that matches your production setup:
>
> - **`http` is the production path** — point `vector_store.url` at your real
>   vector DB / retrieval service (Pinecone, Qdrant, Weaviate, Elasticsearch,
>   pgvector proxy, a custom retriever…). This is what answers "how does my RAG
>   pipeline perform against the store I actually run?"
> - **`json` is for framework testing** — keyword overlap, no embeddings;
>   retrieval is ~free so the benchmark isolates LLM/quality behavior.
> - **`chroma` is only for Chroma-backed setups** — e.g. "how does retrieval
>   latency scale as my local Chroma collection grows?" The stage-timing
>   metric shows Chroma's per-request query time like any other dependency;
>   we are not profiling ChromaDB internals. Skip it (and skip
>   `pip install chromadb`) unless that's your store — it's a lazy import,
>   not a package extra, precisely because most runs don't need it.

#### Real vector stores: Pinecone, Weaviate, Elasticsearch, pgvector

Native vector search needs a **query embedding**, not raw text — the
framework covers this with a dedicated `embed` stage (any
OpenAI-compatible `/embeddings` endpoint: OpenAI, Ollama, LM Studio,
LiteLLM proxy, etc.):

```yaml
pipeline:
  - stage: embed                       # question -> vector
    embedding:
      provider: openai_compat
      model: text-embedding-3-small
      base_url: https://api.openai.com/v1
      api_key_env: OPENAI_API_KEY
  - stage: retrieve
    top_k: 5
    vector_store:
      type: http                       # your store's query endpoint/proxy
      url: https://your-store/search
      api_key_env: STORE_API_KEY
  - stage: generate
    ...
```

The `embed` stage stores the vector on the request context; `retrieve`
then sends **both** `query` (text) and `query_vector` (float list) so the
store can use whichever it supports. Embed token usage is priced under
the `embedding` cost category in `cost_breakdown`, and embed latency
appears as its own `embed` stage/segment in the composition.

Per-store notes:

| Store | What you need |
|---|---|
| Pinecone | A thin proxy/service that accepts `{query_vector, top_k}` and returns `matches`-style hits (or use their inference endpoint for text→vector); native query API is vector-only |
| Weaviate | Proxy translating to GraphQL/REST, or BM25/hybrid text search via `query` |
| Elasticsearch | `_search` endpoint — `match`/BM25 on `query`, or kNN proxy that embeds `query_vector` |
| pgvector | Any HTTP service in front of Postgres (there is no native HTTP API); pass `query_vector` through |
| Chroma | `type: chroma` uses `query_vectors` automatically when an embed stage ran — true vector search, no proxy needed |

If your store already sits behind a company `/search` or `/retrieve`
endpoint returning normalized docs, you need **no embed stage** — plain
`type: http` retrieve is enough (YAML only, no code changes).

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
      # - stage: embed                     # OPTIONAL — real vector stores only
      #   embedding: { provider: openai_compat, model: text-embedding-3-small,
      #                base_url: https://api.openai.com/v1, api_key_env: OPENAI_API_KEY }
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

- **`embed`** — embeds `question` via an OpenAI-compatible `/embeddings`
  endpoint (`embedding:` block: `provider`, `model`, `base_url`,
  `endpoint` override, `api_key_env`, `params`, `timeout_s`). The vector
  lands on `ctx.embedding` for downstream `retrieve` stages. Emits an
  `embed` stage-timing metric; embed tokens are costed under the
  `embedding` category. Optional — skip it for text-search stores.
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
- **`agent`** — tool-calling loop *instead of* `generate`; the model decides
  which tools to call and when to stop. See **Agent pipelines** below.

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

## 2b. Agent pipelines (tool-calling loops)

The `agent` stage is **not** the same thing as a RAG pipeline — it's a
different execution model inside the same `layer: pipeline` target:

| | RAG pipeline (`retrieve → rerank → generate`) | Agent pipeline (`stage: agent`) |
|---|---|---|
| Flow | Deterministic: fetch docs once → one LLM call | Iterative: LLM emits `tool_calls` → engine executes → results fed back → repeat until final answer or `max_iterations` |
| Retrieval | Fixed stage, every request, same `top_k` | A *tool the model chooses* to call — `type: retrieve` tools hit the same vector stores on demand |
| Cost shape | ~1 generation call (+rerank calls) per request | N LLM calls per request (one per step) + tool calls — watch `cost_breakdown.agent` |
| Latency shape | retrieve+rerank+generate segments | `agent_step` + `agent_tool` segments per iteration |
| Quality signal | `CONTEXT_RECALL` + IR metrics (recall@k…) | `AGENT_SUCCESS` task-eval + tool error/loop rates |
| Failure modes | Bad retrieval → bad context | Tool errors, loops, max-iteration exhaustion, cost-cap stops |

They compose: `retrieve → agent` gives the agent pre-fetched context *plus*
tools; an agent with a `type: retrieve` tool does retrieval **on demand**
(agentic RAG) instead of every request paying for it.

### Agent config (`configs/agent_example.yaml`)

```yaml
targets:
  - name: support_agent
    layer: pipeline
    dataset: questions                 # expected_answer drives success eval
    pipeline:
      - stage: agent
        max_iterations: 6              # hard stop on loop length
        max_cost_usd: 0.10             # stop the loop if est. spend exceeds this
        loop_repeat_threshold: 3        # identical tool+args xN -> loop flag
        success:                       # task-success evaluation -> AGENT_SUCCESS
          expected_field: expected_answer   # dataset column compared to final answer
          match: contains              # contains | exact | regex
          on_max_iterations: fail      # fail | pass — exhausting iterations = failure?
          require_tool_use: false      # require >=1 tool call for success
        llm:
          provider: openai             # any llm_api provider works here
          model: gpt-4o
          api_key_env: OPENAI_API_KEY
          params: { temperature: 0.2, max_tokens: 1024 }
        tools:
          # type: http — POST tool args as JSON to an external API
          - name: search_docs
            type: http
            description: Search the internal docs index
            url: http://localhost:8080/search
            method: POST
            api_key_env: SEARCH_API_KEY
            parameters:
              query: { type: string, description: search query }
          # type: function — import path "module.path:callable" (args -> kwargs)
          - name: get_order_status
            type: function
            import_path: "example_tools.orders:get_status"
            description: Look up order status by order id
          # type: retrieve — vector-store tool inside the loop (agentic RAG)
          - name: company_kb
            type: retrieve
            description: Search the company knowledge base
            vector_store: { type: json, path: ../datasets/docs.jsonl }
            top_k: 3
```

Tool types: `http` (external API, SSRF-guarded), `function` (Python import —
requires `security.python_execution: allow` or trusted config), `retrieve`
(same `vector_store` backends as the retrieve stage).

### Agent metrics & thresholds

Report fields per scenario: `agent_runs`, `agent_success_rate`,
`agent_steps_avg`, `agent_tool_calls_avg`, `agent_tool_error_rate`,
`agent_loops_avg`; stage timings `agent_step`/`agent_tool`; all agent LLM
spend lands in `cost_breakdown.<scenario>.agent`.

```yaml
thresholds:
  scenarios:
    agent_load:
      p95_ms: 120000                 # agent e2e is N x llm_api latency — set generously
      max_error_rate: 0.05
      min_agent_success_rate: 0.8    # task success >= 80%
      max_agent_tool_error_rate: 0.1 # tool failures <= 10% of calls
      max_agent_loops_avg: 0.2       # loop detections per run
      stages:
        agent_step: { p95_ms: 30000 }
        agent_tool: { p95_ms: 5000 }
```

### Reading agent results

- **`agent_success_rate` < target** → check `agent_loops_avg` (looping
  without progress), `agent_tool_error_rate` (broken tools), and
  `on_max_iterations` semantics — exhausted iterations may be counted as
  failures by design.
- **Steps avg ≫ expected** → the model is over-calling tools; tighten the
  tool `description`s or lower `max_iterations`.
- **`cost_breakdown.agent` dominates** → each step is a full LLM call;
  fewer/simpler tools or a cheaper `llm.model` for the agent loop helps.
- **Composition** shows `agent_step` vs `agent_tool` median shares — tool
  latency vs LLM latency split per request.

## 3. Run the test

```bash
# install (once)
pip install -e .
# optional, only for chroma vector stores (not a package extra):
pip install chromadb

# validate config before spending tokens
llm-perf validate -c configs/rag_example.yaml
llm-perf validate -c configs/agent_example.yaml   # agent pipeline variant

# quick smoke (5 users, 2 minutes)
llm-perf run -c configs/rag_example.yaml --headless \
  --users 5 --spawn-rate 1 --duration 2m

# agent pipeline — same commands, agent_example.yaml
llm-perf run -c configs/agent_example.yaml --headless \
  --users 3 --spawn-rate 1 --duration 3m

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
`demo/rag_embed_dryrun.yaml` additionally exercises the `embed` stage +
HTTP vector store (`query_vector` → mock `/retrieve`) with no credentials.

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
