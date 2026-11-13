# Workload profiles & end-to-end testing with llm-perf

This guide covers **`load:` configuration** — how to shape the traffic
(ramp-up → steady → ramp-down, step profiles, spikes, soak) — and the
**end-to-end flow** from config to gated CI result. It applies to every
layer (`llm_api`, `pipeline`, `app_endpoint`, `app_object`).

## 1. How load works

Two ways to drive concurrency:

- **Flat load** — no `profile`, no `stages`: Locust spawns `users` at
  `spawn_rate` and holds for `duration`. Simplest; good for quick checks.
- **Profiled load** — `load.profile` set (or explicit `load.stages`): a
  Locust `LoadTestShape` walks a **phase schedule** built by
  `llm_perf/workload.py`. Each phase is `(target_users, spawn_rate,
  duration_s, label)`; the shape prints `phase <label>: users=N …` as it
  transitions, and the full schedule lands in `report.phases`.

```yaml
load:
  users: 20            # peak concurrency (step profiles: final level)
  spawn_rate: 4        # users spawned per second
  profile: load        # see table below
  duration: 5m         # steady-state hold (per-step hold for step profiles)
  ramp_up: 2m          # optional — sensible defaults per profile
  ramp_down: 1m
  run_time: 30m        # optional hard cap on total run time
```

CLI overrides (no YAML edits):

```bash
llm-perf run -c configs/profiles_example.yaml --headless --profile stress
llm-perf run -c configs/openai_example.yaml --headless --users 50 --spawn-rate 10 --duration 10m
llm-perf run -c configs/x.yaml --headless --ramp-up 2m --ramp-down 1m
```

`--layers` / `--scenarios` filter which targets/scenarios run;
`--profile` accepts `smoke|baseline|load|stress|spike|soak|scalability|
capacity|custom`.

## 2. The nine profiles

| Profile | Phase shape | Defaults (overridable) | Use it for |
|---|---|---|---|
| `smoke` | ramp → hold → ramp_down | hold 120s, ramps 30s/15s | Config sanity, CI smoke gate |
| `baseline` | ramp → hold → ramp_down | hold 600s, ramps 120s/60s | Reference run for regression diffs |
| `load` | ramp → hold → ramp_down | ramp = `users/spawn_rate`, hold = `duration` | Expected production load |
| `stress` | N equal steps up to `users`, hold each | `steps: 4`, hold = `step_duration` or `duration` | Find the breaking point gradually |
| `spike` | spike up → hold → drop to `baseline_users` → down | baseline = `users/10`, baseline hold 120s | Sudden-traffic resilience |
| `soak` | ramp → long hold → ramp_down | hold **4h**, ramps 300s/300s | Leaks, memory growth, throttling over time |
| `scalability` | step up by `step_users`, hold each + final steady | `step_users` default `users/5`, hold = `step_duration`/`duration` | Measure rps/latency vs concurrency curve |
| `capacity` | same steps as scalability, **stops early on error breach** | `capacity_max_error_rate: 0.05` | "How much load before it falls over?" |
| `custom` | explicit `load.stages` list | — | Arbitrary schedules (diurnal patterns, canary ramps) |

Step-profile knobs: `steps` (stress), `step_users` + `step_duration`
(scalability/capacity), `baseline_users` + `baseline_duration` (spike),
`capacity_max_error_rate` (capacity — the shape prints
`capacity profile: error-rate breach, stopping early` and ends the run).

### Custom stages

```yaml
load:
  profile: custom          # or omit profile — stages imply custom
  stages:
    - { users: 5,  spawn_rate: 1, duration: 2m }   # warm-up
    - { users: 20, spawn_rate: 4, duration: 5m }   # business-hours level
    - { users: 50, spawn_rate: 10, duration: 3m }  # peak
    - { users: 10, spawn_rate: 2, duration: 2m }   # decay
  ramp_down: 1m
```

A final `ramp_down` phase to 0 users is appended automatically
(`load.ramp_down`, default 30s).

### Think time & scenario mix

- `scenarios[].think_time: [min, max]` — seconds a virtual user pauses
  between requests (uniform; **default 0.5–2.0s when unset**). Keep it
  realistic: `think_time` dominates rps at high user counts.
- `scenarios[].weight` — relative share when multiple scenarios target
  the same/different layers in one run.
- Target-level `vars:` pools feed `{var}` placeholders in prompts and
  `request_template`s — workload *content* variety, orthogonal to load shape.

## 3. End-to-end flow

```
config → validate → smoke → profiled runs → report → thresholds/regression → CI gate
```

1. **Write the config** — pick layer targets, dataset/prompts, scenario
   weights + think_time, `load:` block (start flat or `profile: smoke`),
   `thresholds:`, `report:` (pricing, history, baseline).
2. **Validate** — `llm-perf validate -c cfg.yaml` (schema + env vars +
   security audit, no traffic).
3. **Smoke** — `--profile smoke` or `-u 2 -t 30s`: proves the pipeline
   end-to-end cheaply before committing hours.
4. **Baseline** — `--profile baseline` (or `load` for prod-shaped load)
   against the *current* build; this becomes the comparison anchor.
5. **Stress/capacity** — `--profile stress` or `capacity` to find the
   knee; watch `phases[]` + per-phase console labels to correlate
   latency/error jumps with concurrency levels.
6. **Soak** (optional, scheduled) — `--profile soak` overnight for
   memory/leak/throttle drift; compare `resources` sections across runs.
7. **Read the report** — latency percentiles, rps, error rate,
   `cost_breakdown`, quality sampling counts, `phases[]` schedule,
   `resources` (generator headroom), threshold verdicts.
8. **Regression check** — `report.baseline: previous|designated|<run.json>`
   + `report.regression` compares against history; `manifest` provenance
   (git commit, config/dataset hashes) flags non-comparable runs.
9. **Gate CI** — `thresholds.on_fail: fail` and/or
   `regression.on_fail: fail` → non-zero exit on SLO breach.

### Distributed runs

Profiles work unchanged with master/workers — the shape runs on the
master, workers execute tasks:

```bash
llm-perf run -c cfg.yaml --master=0.0.0.0 --workers 4 --profile load
llm-perf run -c cfg.yaml --worker --master <host>
```

`load.users` is the **total** across workers. Datasets shard
deterministically — verify coverage first:

```bash
llm-perf dataset coverage -c cfg.yaml -w 4
```

## 4. Reading profiled results

- **`report.phases[]`** — the executed schedule
  (`{label, target_users, spawn_rate, duration_s}`); console prints the
  same transitions live.
- **Metrics are run-aggregated** — percentiles/rps cover the whole run,
  not per-phase slices. To compare phases, run separate configs or use
  `capacity`/`scalability` step structure and correlate the console
  phase labels with the HTML report's timeline charts.
- **`capacity` early stop** — the run ends when observed error rate
  exceeds `capacity_max_error_rate`; the report still contains full
  stats up to that point. Treat "stopped early" itself as the finding.
- **`resources`** — `collect_resources: true` records load-generator
  CPU/mem; if generator CPU saturates, latency rises are *harness*
  artifacts, not target regressions — scale workers or reduce users.
- **History** — every run is archived under `report.output_dir/history/`
  when `report.history: true`; `report.baseline` picks the comparison
  anchor (`previous` run, `designated` path, or `none`).

## 5. Recommended profile ladder

| Stage | Profile | Why |
|---|---|---|
| PR / pre-merge | `smoke` (or flat `-u 5 -t 2m`) + `on_fail: fail` thresholds | Fast gate, catches breakage |
| Release candidate | `baseline` + `load` | Reference + prod-shaped evidence |
| Performance investigation | `stress` → `capacity` | Locate saturation point |
| Production confidence | `soak` (scheduled) | Long-horizon stability |
| Regression tracking | any + `report.baseline`/`regression` | Automated delta vs last good run |

## 6. Example: full annotated run

`configs/profiles_example.yaml` is a complete working example — an
`llm_api` streaming target with retries, `profile: load`, rich thresholds
(TTFT/TPOT/ITL/gen/queue percentiles, quality, cost-per-success), pricing,
history + regression, and an optional app-side `/metrics` scrape.

```bash
# validate
llm-perf validate -c configs/profiles_example.yaml --skip-connectivity

# CI-style smoke
llm-perf run -c configs/profiles_example.yaml --profile smoke --headless

# production-shaped load
llm-perf run -c configs/profiles_example.yaml --profile load --headless \
  --users 30 --duration 10m

# find capacity
llm-perf run -c configs/profiles_example.yaml --profile capacity --headless \
  --users 100 --step-users 10 --step-duration 3m
```

No credentials? Use the mock-backed demo instead:

```bash
python demo/mock_server.py &
llm-perf run -c demo/all_layers.yaml --headless --profile smoke
llm-perf run -c demo/all_layers.yaml --headless --profile stress --users 12
```
