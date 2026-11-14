# Benchmark-driven optimization

`llm-perf optimize` turns measured pipeline bottlenecks into controlled
experiments. It runs the current configuration plus bounded variants under the
same workload, applies quality/reliability/cost constraints, and selects the
candidate with the best objective value.

## What it is used for

Use optimization when a report shows that end-to-end latency is too high and
you want to compare safe configuration changes without manually editing the
YAML for every experiment.

Typical questions it answers:

- Does `retrieve.top_k: 3` produce a better query-to-response p95 than `top_k: 8`?
- Does a lower `max_tokens` value reduce generation latency while preserving quality?
- Does a smaller agent `max_iterations` limit improve p95 without reducing task success?
- Does a cheaper/faster model still satisfy quality and error-rate constraints?

## Why it exists

Percentile reports and bottleneck findings identify *where* latency is spent,
but they do not prove which configuration is best. This feature closes that
loop by measuring candidate configurations directly instead of relying on
intuition.

The optimizer is deliberately **benchmark-driven**, not a runtime mutator:
every candidate is evaluated under the same dataset, load shape, and scenario,
so latency changes can be attributed to the parameter being tested.

## Supported optimization paths

For `pipeline` targets, use `stage_index` to address a stage and `path` to
select a bounded field:

| Stage/path prefix | Examples |
|---|---|
| `top_k` | retrieval depth |
| `method` | rerank method |
| `template` | alternate prompt template path |
| `max_iterations` | agent loop length |
| `max_cost_usd` | agent cost cap |
| `loop_repeat_threshold` | repeated tool-call loop detection |
| `llm.model` | stage LLM model |
| `llm.params.*` | `max_tokens`, `temperature`, provider params |
| `embedding.model` | embedding model |
| `embedding.params.*` | embedding provider params |

For non-pipeline targets, target-level paths are intentionally narrower:

| Target type | Supported paths |
|---|---|
| `llm_api` | `params.*`, `stream`, `stream_mode`, `timeout_s` |
| `app_endpoint` | `timeout_s` |
| `app_object` | `call`, `init_args.*` |

Credentials, URLs, datasets, security settings, tool definitions, load shape,
and report/history settings are not optimization parameters. This prevents an
optimization run from bypassing guardrails or changing the workload being
measured.

## Configuration

```yaml
optimization:
  enabled: true
  scenario: rag_load
  objective: p95_ms
  strategy: grid             # grid | random
  max_trials: 8              # includes the baseline candidate
  min_requests: 30           # trials below this cannot win
  seed: 7                    # random strategy ordering

  constraints:
    max_error_rate: 0.02
    min_quality_score: 0.70
    max_cost_per_success_usd: 0.02
    min_agent_success_rate: 0.80
    stages:
      retrieve: { p95_ms: 2000 }
      generate: { p95_ms: 40000 }

  parameters:
    - target: rag_pipeline
      stage_index: 0
      path: top_k
      values: [3, 5, 8]

    - target: rag_pipeline
      stage_index: 2
      path: llm.params.max_tokens
      values: [256, 512]
```

`stage_index` is the zero-based position in `targets[].pipeline`. In a
`retrieve → rerank → generate` pipeline, index `0` is `retrieve`, `1` is
`rerank`, and `2` is `generate`.

## Running

```bash
# list the baseline + candidates without sending traffic
llm-perf optimize -c configs/my_pipeline.yaml --dry-run

# execute the experiment matrix (each trial runs headless)
llm-perf optimize -c configs/my_pipeline.yaml

# CLI overrides for quick sweeps
llm-perf optimize -c configs/my_pipeline.yaml \
  --strategy random --trials 6 --objective p95_ms
```

The command executes each candidate through the normal `llm-perf run` path in
a separate subprocess. Trial reports therefore use the same request execution,
metrics, thresholds, tracing, and reporting code as a normal run.

## Outputs

Optimization artifacts are written under
`report.output_dir/optimization/<timestamp>/` unless `--output-dir` is set:

| File | Contents |
|---|---|
| `trial-NN.yaml` | Effective config used for that candidate |
| `trial-NN/llm-perf-*.json` | Normal run report for the candidate |
| `optimization-report.json` | Every trial's parameters, score, metrics, violations, and winner |
| `optimized.yaml` | Winning configuration with optimization disabled and normal report settings restored |

Trial runs have history and regression comparison disabled internally so the
experiment does not accidentally become the production baseline.

## Selection behavior

- Candidate `0` is always the unmodified **baseline** configuration.
- `max_trials` includes that baseline. For `max_trials: 8`, the run evaluates
  baseline plus up to seven variants.
- A candidate is feasible only when:
  - the run produces a report,
  - the optimized scenario meets `min_requests`,
  - the objective metric exists, and
  - every configured constraint passes.
- The feasible candidate with the lowest objective value wins.
- If no candidate is feasible, the command exits non-zero and no winner is
  selected.

## Report recommendations

Every normal run report also includes `analysis.optimization`, a structured
list generated from the measured bottleneck. For example, a dominant
`generate` segment maps to generation-tuning paths such as `llm.model` and
`llm.params.max_tokens`; dominant `agent_tool` latency points to tool latency
and loop controls.

These recommendations are evidence for choosing `optimization.parameters`;
they do not modify the pipeline by themselves.
