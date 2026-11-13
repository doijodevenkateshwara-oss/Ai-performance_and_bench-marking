# Performance testing LangChain / LlamaIndex / custom Python objects with llm-perf

This guide covers the **`app_object` layer** — benchmarking your
application *in-process*: a LangChain chain, a LlamaIndex query engine, or
any custom Python object, invoked directly per request with **no HTTP hop**.
You measure the framework/object itself — retrieval, prompt assembly,
model calls, post-processing — exactly as your code runs it.

Use this layer when the code under test is a Python object you own.
For a deployed HTTP/gRPC service use `app_endpoint`; for vendor LLM APIs
use `llm_api` (see the sibling guides).

## 1. What gets measured

| Signal | Where it appears | Why it matters |
|---|---|---|
| End-to-end call latency (p50/p90/p95/p99) | `scenarios[].p50_ms…p99_ms` | Full in-process call time — your code + any LLM calls it makes |
| Throughput (`rps`) | `scenarios[].rps` | How many invocations/sec the object sustains |
| Error rate | `error_rate`, `failures`, exception text in report | Exceptions from your code surface as request failures |
| Active users / think time | `load`, scenario `think_time` | Concurrency model — greenlets share one process |
| Resource usage | `resources` | In-process runs share CPU/memory with the load generator — watch generator headroom carefully |
| Quality (heuristics + judge) | `quality_score_avg`, `quality_pass_rate` | Works on extracted response text |
| Threshold verdicts | `thresholds.results[]` | pass / fail / not_evaluated |

What you do **not** get: per-stage breakdown *inside* your object (the
layer sees one call), token/cost accounting (the object owns its own LLM
calls — instrument them separately or accept latency-only evidence), and
network-level metrics. If your object wraps an LLM, its internal latency
is folded into e2e — use `app_endpoint` against the same app's HTTP front
if you want the transport split.

## 2. The contract: `python_path` + `call`

```yaml
targets:
  - name: my_chain
    layer: app_object                  # inferred if python_path/framework present
    framework: langchain               # langchain | llamaindex | generic
    python_path: "my_app.chains:build_chain"   # module.path:factory
    call: ainvoke                      # method invoked per request
    init_args: { model: gpt-4o-mini }  # passed to the factory
    dataset: prompts_ds
    prompts: ["Say hi to {name}"]
    vars: { name: ["alice", "bob"] }
```

**`python_path` resolution** (`module.path:callable`):

- The module must be **importable** — install your package
  (`pip install -e ../my-app`) or set `PYTHONPATH=/path/to/my-app` before
  running. The framework does not add the config directory to `sys.path`.
- Resolution rules (`_import_factory`):
  - **plain function** → called once per worker as `factory(**init_args)`;
    the return value is the app object.
  - **anything else** (class, instance, async function) → used *directly*
    as the app — `init_args` is **not** applied. To pass constructor args,
    wrap them in a factory function.
- The app is imported/instantiated **once per worker process** and reused
  for every request — model clients, retrievers, and connections are
  *not* rebuilt per call (that would measure import time, not runtime).

**`call`** — the method invoked on the app each request:

| Value | Typical use |
|---|---|
| `ainvoke` / `invoke` | LangChain `Runnable` / custom sync-async methods |
| `query` / `aquery` | LlamaIndex query engines (always positional `fn(prompt)`) |
| any other string | custom method name, e.g. `call: run` |

Argument convention: `query`/`aquery` (and `framework: llamaindex`)
receive the prompt **positionally**; other methods get `input=prompt`
when the signature declares an `input` parameter, else positional.
Awaitable results are awaited on a per-greenlet event loop (greenlets
share the OS thread, so each gets its own asyncio loop driven
cooperatively under gevent's patched selectors).

**Response extraction** (`_extract_text`, best-effort order): plain
`str` → `.content` attribute (str, or list of `{text}` parts) → dict keys
`output_text`/`text`/`content`/`response`/`answer` → LangChain-style
`generations[0][0].text` → `str(result)` fallback. If your object returns
an unusual type, return a `str` or a dict with a `text`/`content` key to
get clean quality scoring.

## 3. Framework-specific notes

### LangChain

```yaml
- name: support_chain
  layer: app_object
  framework: langchain
  python_path: "my_app.rag:build_chain"     # returns a compiled chain
  call: ainvoke
  init_args: { top_k: 5 }
```

`ainvoke`/`invoke` map to `chain.ainvoke(input=prompt)` /
`chain.invoke(input=prompt)` — Runnable's `input` kwarg path. Return
values are usually `AIMessage` (`.content`) or `dict` — both extract
cleanly. For chains returning `dict` with intermediate steps, only the
final text reaches quality checks.

### LlamaIndex

```yaml
- name: qa_engine
  layer: app_object
  framework: llamaindex
  python_path: "my_app.index:build_engine"  # returns QueryEngine
  call: query                               # or aquery
```

`query`/`aquery` are invoked positionally (`engine.query(prompt)`),
matching LlamaIndex's API. `QueryEngine` results have `.response` —
covered by dict/attr extraction; if yours returns a custom `Response`
object, ensure it stringifies usefully or exposes `.response`.

### Custom Python object

Any object works — set `framework: generic` (or omit; inference picks
`app_object` from `python_path`). Implement `invoke`/`ainvoke`/your own
method name and return `str`/dict/`AIMessage`-like results.

## 4. Install extras

```bash
pip install -e ".[langchain]"     # langchain + langchain-core
pip install -e ".[llamaindex]"    # llama-index
```

`generic` needs nothing extra. Your app's own dependencies (vector
stores, model SDKs) must be installed in the same environment.

## 5. Security: `python_execution` gate

`python_path` imports and executes **your code with full process
privileges** — the framework gates it:

```yaml
security:
  python_execution: warn    # allow | warn | off  (default: warn)
```

- `warn` — runs, prints a one-time loud warning
- `allow` — runs silently (CI/prod acknowledgement)
- `off` — `PermissionError` before any import

`audit_config` also flags every `app_object` target in the
untrusted-config warning. Only benchmark code you trust — a malicious
`python_path` is arbitrary code execution.

## 6. Full example config

Based on the working demo (`demo/dummy_app.py` + `demo/all_layers.yaml`):

```yaml
observability:
  otel: { enabled: true, endpoint: http://localhost:4317, service_name: llm-perf-appobj }

datasets:
  prompts_ds:
    source: ../datasets/prompts.jsonl    # {"prompt": "...", "expected": "..."}
    format: jsonl
    sample: { strategy: random, count: 100, seed: 3 }

security:
  python_execution: allow

targets:
  - name: dummy_app
    layer: app_object
    framework: generic
    python_path: demo.dummy_app:build    # factory(**init_args) -> DummyApp
    call: invoke
    init_args: { prefix: "evidence-app" }
    dataset: prompts_ds

scenarios:
  - name: object_load
    target: dummy_app
    weight: 100
    think_time: [0.05, 0.2]
    quality:
      sample_rate: 0.5
      heuristics: { non_empty: true, min_chars: 5 }
      judge:
        provider: openai
        model: gpt-4o-mini
        api_key_env: OPENAI_API_KEY
        rubric: "Score correctness 1-5."
        scale: [1, 5]

load:
  profile: smoke
  users: 5
  spawn_rate: 5
  duration: 2m

thresholds:
  on_fail: fail
  scenarios:
    object_load:
      p95_ms: 5000
      max_error_rate: 0.02
      min_quality_score: 0.6

report:
  formats: [csv, json, html]
  output_dir: results/app-object
  collect_resources: true
```

## 7. How to run

```bash
# from repo root; your package must be importable (pip install -e or PYTHONPATH)
export PYTHONPATH=/path/to/your-app:${PYTHONPATH:-}

# validate
llm-perf validate -c configs/app_object_example.yaml

# smoke first — 2 users, 30s
llm-perf run -c configs/app_object_example.yaml --headless -u 2 -t 30s

# real run / profile
llm-perf run -c configs/app_object_example.yaml --headless --profile load

# filter a mixed config to just object targets
llm-perf run -c configs/all_layers.yaml --layers app_object --headless

# distributed — each worker imports the app independently; datasets shard
llm-perf run -c configs/app_object_example.yaml --master=0.0.0.0 --workers 4
```

Smoke-test checklist before a long run: one manual invocation of your
factory + call path in a Python REPL to confirm the object builds and
returns extractable text.

## 8. View results

Same report pipeline as every layer — `llm-perf-<ts>.{csv,json,html}`:

```jsonc
{
  "scenarios": [{
    "scenario": "object_load",
    "requests": 940, "failures": 0, "error_rate": 0.0,
    "rps": 7.8,
    "avg_ms": 112.4, "p50_ms": 98.0, "p95_ms": 210.3, "p99_ms": 402.1,
    "quality_score_avg": 0.94, "quality_evaluated": 470,
    "quality_judged": 235, "quality_judge_rate": 0.5
  }],
  "thresholds": { "results": [ {"name": "scenario:object_load.p95_ms",
                                 "passed": true, "actual": 210.3, "threshold": 5000} ] }
}
```

Provider label in metrics/traces is `app_<framework>` (e.g.
`app_langchain`, `app_generic`), so Grafana/Jaeger can split object runs
from endpoint/API runs.

## 9. Validate results

| Check | Healthy | Investigate when |
|---|---|---|
| `error_rate` | 0 | Exceptions from your object — read `failures` + span `error` attrs; a bad `python_path`/`call` shows as 100% errors immediately |
| `p95_ms` | stable across users | Rising with concurrency → shared state/locks inside your object, or downstream LLM rate limits |
| `rps` | scales then plateaus | In-process: GIL/event-loop saturation; check `resources` — the *generator* may be the ceiling, not your object |
| `quality_score_avg` | ≥ threshold | Extraction may be grabbing the wrong field — check `_extract_text` order vs your return type |
| Latency vs `app_endpoint` of same app | in-process ≤ endpoint | If in-process is *slower*, suspect event-loop contention or per-call object rebuild |

Important caveat for interpretation: **all greenlets share one process**
— each greenlet drives its own event loop for async app calls, but they
still contend for the same CPU/GIL. Results measure your object under
cooperative concurrency, not multi-process isolation. For capacity
planning of a *deployed* service, corroborate with `app_endpoint` runs.

## 10. End-to-end flow

1. **Wrap/choose the object** — ensure a factory (`module:build`) exists;
   add one if needed (10-line shim is fine).
2. **Install deps** — `pip install -e ".[langchain]"` or your package on
   `PYTHONPATH`.
3. **Prepare data** — dataset with `prompt`/`expected` columns, or
   `prompts:` + `vars:`.
4. **Write config** — §6 template; set `security.python_execution: allow`
   deliberately.
5. **Smoke** — `-u 2 -t 30s`; confirm 0 errors and sane latency.
6. **Validate** — `llm-perf validate -c …`.
7. **Run** — profile or explicit `--users/--spawn-rate/--duration`;
   `collect_resources: true` to check generator headroom.
8. **Read report** — latency percentiles, error rate, quality sampling
   counts + CI, threshold verdicts.
9. **Compare** — `report.baseline`/`regression` vs previous runs; manifest
   provenance (git commit + config hash) flags non-comparable runs.
10. **Gate CI** — `thresholds.on_fail: fail` once the object is stable.

## 11. Diagnostics

| Symptom | Likely cause | Where to look |
|---|---|---|
| 100% errors instantly | `python_path` not importable, `call` method missing, or `python_execution: off` | First failure message in report; test import manually |
| Works at u=1, fails at u=N | Shared mutable state, unbounded queues, downstream rate limit | Exception text; server-side logs |
| Latency climbs linearly with users | GIL-bound CPU work in your object | `resources` CPU% on generator; profile the call |
| Async calls slower than sync equivalent | Event-loop contention across greenlets | Try fewer users or `call: invoke` |
| Quality score low but answers look right | `_extract_text` picked wrong field | Return `str`/`{"text": ...}` from your method |
| `cost_usd` = 0 | Expected — object owns its LLM spend | Instrument inside your app or use llm_api layer for cost |

## 12. Security notes

- `python_path` is **arbitrary code execution** — the
  `security.python_execution` gate (`allow`/`warn`/`off`) exists
  specifically for this; `off` hard-blocks before import.
- The imported module runs with the framework's full privileges —
  filesystem, network, env vars. Benchmark only trusted code.
- `init_args` values come from YAML — keep secrets out; read credentials
  from env vars inside your factory instead.
- Datasets still go through the standard caps (`dataset_max_bytes`,
  `dataset_max_rows`, `dataset_timeout_s`) and URL policy for remote
  sources.
- OTel spans carry `output_chars` only — response text is not exported,
  which is the right default for potentially sensitive app output.
