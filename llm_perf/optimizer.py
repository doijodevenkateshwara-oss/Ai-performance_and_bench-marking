"""Benchmark-driven optimization for declarative pipeline and API targets.

The optimizer generates bounded config variants, executes each variant under
an identical workload, applies quality/reliability/cost constraints, and
selects the candidate with the best objective value.


Author: Venkateshwara Doijode
"""

from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import (
    AppEndpointTarget,
    AppObjectTarget,
    LlmApiTarget,
    OptimizationConfig,
    OptimizationParameter,
    PerfConfig,
    PipelineTarget,
    ThresholdRule,
)

_STAGE_EXACT_PATHS = {
    "top_k",
    "method",
    "template",
    "max_iterations",
    "max_cost_usd",
    "loop_repeat_threshold",
    "llm.model",
    "embedding.model",
}
_STAGE_PREFIX_PATHS = ("llm.params.", "embedding.params.")
_LLM_EXACT_PATHS = {"stream", "stream_mode", "timeout_s"}
_LLM_PREFIX_PATHS = ("params.",)
_ENDPOINT_EXACT_PATHS = {"timeout_s"}
_APP_OBJECT_EXACT_PATHS = {"call"}
_APP_OBJECT_PREFIX_PATHS = ("init_args.",)

_METRIC_FIELDS = (
    "requests",
    "failures",
    "error_rate",
    "rps",
    "avg_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "ttft_p95_ms",
    "tpot_p95_ms",
    "itl_p95_ms",
    "gen_duration_p95_ms",
    "queue_wait_p95_ms",
    "tokens_per_sec_avg",
    "quality_score_avg",
    "quality_pass_rate",
    "context_recall_avg",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg",
    "cost_usd",
    "cost_per_success_usd",
    "agent_runs",
    "agent_success_rate",
    "agent_steps_avg",
    "agent_tool_calls_avg",
    "agent_tool_error_rate",
    "agent_loops_avg",
)

# ThresholdRule field -> (report field, comparison direction). All rules are
# feasibility constraints; a candidate cannot win while violating them.
_CONSTRAINT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("p50_ms", "p50_ms", "<="),
    ("p90_ms", "p90_ms", "<="),
    ("p95_ms", "p95_ms", "<="),
    ("p99_ms", "p99_ms", "<="),
    ("ttft_p95_ms", "ttft_p95_ms", "<="),
    ("tpot_p95_ms", "tpot_p95_ms", "<="),
    ("itl_p95_ms", "itl_p95_ms", "<="),
    ("gen_p95_ms", "gen_duration_p95_ms", "<="),
    ("queue_p95_ms", "queue_wait_p95_ms", "<="),
    ("max_error_rate", "error_rate", "<="),
    ("max_retries", "retries_total", "<="),
    ("max_timeouts", "timeouts", "<="),
    ("min_throughput_rps", "rps", ">="),
    ("min_quality_score", "quality_score_avg", ">="),
    ("min_quality_pass_rate", "quality_pass_rate", ">="),
    ("min_context_recall", "context_recall_avg", ">="),
    ("min_recall_at_k", "recall_at_k", ">="),
    ("min_precision_at_k", "precision_at_k", ">="),
    ("min_mrr", "mrr", ">="),
    ("min_ndcg", "ndcg", ">="),
    ("max_cost_usd", "cost_usd", "<="),
    ("max_cost_per_success_usd", "cost_per_success_usd", "<="),
    ("min_agent_success_rate", "agent_success_rate", ">="),
    ("max_agent_tool_error_rate", "agent_tool_error_rate", "<="),
    ("max_agent_loops_avg", "agent_loops_avg", "<="),
)


@dataclass
class Candidate:
    """One executable optimization candidate."""

    index: int
    label: str
    values: dict[str, Any]
    config: PerfConfig
    report_path: Path | None = None
    returncode: int | None = None
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        """Return the JSON-serializable trial summary."""
        return {
            "trial": self.index,
            "label": self.label,
            "values": self.values,
            "returncode": self.returncode,
            "report_path": str(self.report_path) if self.report_path else None,
            "feasible": self.feasible,
            "violations": self.violations,
            "score": self.score,
            "metrics": self.metrics,
        }


def candidate_configs(config: PerfConfig) -> list[Candidate]:
    """Build the baseline candidate plus bounded parameter combinations."""
    opt = config.optimization
    _validate_optimization(config, opt)
    candidates = [
        Candidate(
            index=0,
            label="baseline",
            values={},
            config=config.model_copy(deep=True),
        )
    ]
    value_lists = [p.values for p in opt.parameters]
    combos = list(itertools.product(*value_lists))
    if opt.strategy == "random":
        rng = random.Random(opt.seed)
        rng.shuffle(combos)
    combos = combos[: max(opt.max_trials - 1, 0)]

    for idx, combo in enumerate(combos, start=1):
        trial_cfg = config.model_copy(deep=True)
        values = {}
        for param, value in zip(opt.parameters, combo, strict=True):
            key = _parameter_key(param)
            values[key] = value
            _apply_parameter(trial_cfg, param, value)
        # Revalidate after mutation so bad values fail before traffic is sent.
        trial_cfg = PerfConfig.model_validate(trial_cfg.model_dump(mode="python", by_alias=True))
        candidates.append(
            Candidate(
                index=idx,
                label=", ".join(f"{k}={v!r}" for k, v in values.items()),
                values=values,
                config=trial_cfg,
            )
        )
    return candidates


def scenario_name(config: PerfConfig) -> str:
    """Return the scenario optimized by this config."""
    if config.optimization.scenario:
        return config.optimization.scenario
    if not config.scenarios:
        raise ValueError("optimization requires at least one scenario")
    return config.scenarios[0].name


def score_report(
    report: dict,
    scenario: str,
    objective: str,
    constraints: ThresholdRule | None = None,
    min_requests: int = 1,
) -> dict:
    """Evaluate one run report for objective score and feasibility."""
    sc = next((s for s in report.get("scenarios", []) if s.get("scenario") == scenario), None)
    violations: list[str] = []
    metrics: dict[str, Any] = {}
    score = None
    if sc is None:
        violations.append(f"scenario report missing: {scenario}")
    else:
        metrics = {k: sc.get(k) for k in _METRIC_FIELDS if sc.get(k) is not None}
        requests = sc.get("requests") or 0
        if requests < min_requests:
            violations.append(f"insufficient requests: {requests} < {min_requests}")
        score = sc.get(objective)
        if score is None:
            violations.append(f"objective not evaluated: {objective}")
        if constraints:
            violations.extend(_constraint_violations(constraints, sc, report, scenario))
    return {
        "feasible": not violations,
        "violations": violations,
        "score": score,
        "metrics": metrics,
    }


def run_optimization(
    config_path: str | Path,
    *,
    config: PerfConfig | None = None,
    output_dir: str | Path | None = None,
    runner: Callable[[Path], int] | None = None,
) -> dict:
    """Execute optimization trials and write the optimization artifacts."""
    config_path = Path(config_path)
    base = config or PerfConfig.from_yaml(config_path)
    candidates = candidate_configs(base)
    scenario = scenario_name(base)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(output_dir) if output_dir else Path(base.report.output_dir) / "optimization" / stamp
    root.mkdir(parents=True, exist_ok=True)

    for cand in candidates:
        trial_cfg = _trial_config(cand.config, root / f"trial-{cand.index:02d}")
        trial_path = root / f"trial-{cand.index:02d}.yaml"
        _write_config(trial_cfg, trial_path)
        cand.returncode = (runner or _run_cli)(trial_path)
        cand.report_path = _latest_report(root / f"trial-{cand.index:02d}")
        if cand.returncode != 0:
            cand.violations.append(f"run exited {cand.returncode}")
        if cand.report_path is None:
            cand.violations.append("report not produced")
            cand.feasible = False
            continue
        try:
            report = json.loads(cand.report_path.read_text())
        except Exception as e:
            cand.violations.append(f"report unreadable: {e}")
            cand.feasible = False
            continue
        scored = score_report(
            report,
            scenario,
            base.optimization.objective,
            constraints=base.optimization.constraints,
            min_requests=base.optimization.min_requests,
        )
        cand.score = scored["score"]
        cand.metrics = scored["metrics"]
        cand.violations.extend(scored["violations"])
        cand.feasible = not cand.violations

    winner = min(
        (c for c in candidates if c.feasible and c.score is not None),
        key=lambda c: c.score,
        default=None,
    )
    optimized_path = None
    if winner:
        optimized_path = root / "optimized.yaml"
        _write_config(_optimized_config(winner.config, base), optimized_path)

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path),
        "scenario": scenario,
        "objective": base.optimization.objective,
        "strategy": base.optimization.strategy,
        "trials": [c.summary() for c in candidates],
        "winner": winner.summary() if winner else None,
        "optimized_config": str(optimized_path) if optimized_path else None,
        "output_dir": str(root),
    }
    report_path = root / "optimization-report.json"
    report_path.write_text(json.dumps(result, indent=2, default=str))
    result["report_path"] = str(report_path)
    return result


def _validate_optimization(config: PerfConfig, opt: OptimizationConfig) -> None:
    """Validate target/scenario references and mutation paths."""
    if not opt.enabled:
        raise ValueError("optimization.enabled must be true")
    if not opt.parameters:
        raise ValueError("optimization.parameters must define at least one parameter")
    names = {s.name for s in config.scenarios}
    if opt.scenario and opt.scenario not in names:
        raise ValueError(f"optimization scenario not found: {opt.scenario}")
    scenario = opt.scenario or (config.scenarios[0].name if config.scenarios else None)
    if scenario is None:
        raise ValueError("optimization requires at least one scenario")
    target_names = {t.name for t in config.targets}
    for param in opt.parameters:
        if param.target not in target_names:
            raise ValueError(f"optimization parameter target not found: {param.target}")
        target = next(t for t in config.targets if t.name == param.target)
        _validate_parameter(target, param)


def _validate_parameter(target, param: OptimizationParameter) -> None:
    """Reject unsafe or unsupported mutation paths."""
    path = param.path
    if param.stage_index is not None:
        if not isinstance(target, PipelineTarget):
            raise ValueError(f"{param.target}: stage_index requires a pipeline target")
        if param.stage_index >= len(target.pipeline):
            raise ValueError(f"{param.target}: stage_index out of range: {param.stage_index}")
        if path in _STAGE_EXACT_PATHS or path.startswith(_STAGE_PREFIX_PATHS):
            return
        raise ValueError(f"unsupported pipeline optimization path: {path}")

    if isinstance(target, LlmApiTarget):
        if path in _LLM_EXACT_PATHS or path.startswith(_LLM_PREFIX_PATHS):
            return
    elif isinstance(target, AppEndpointTarget):
        if path in _ENDPOINT_EXACT_PATHS:
            return
    elif isinstance(target, AppObjectTarget) and (
        path in _APP_OBJECT_EXACT_PATHS or path.startswith(_APP_OBJECT_PREFIX_PATHS)
    ):
        return
    raise ValueError(f"unsupported optimization path for {param.target}: {path}")


def _apply_parameter(config: PerfConfig, param: OptimizationParameter, value: Any) -> None:
    """Apply one validated parameter value to a config deep copy."""
    target = next(t for t in config.targets if t.name == param.target)
    obj = target.pipeline[param.stage_index] if param.stage_index is not None else target
    _set_path(obj, param.path, value)


def _set_path(root: Any, path: str, value: Any) -> None:
    """Set a dot path on a Pydantic model or nested dict."""
    parts = path.split(".")
    cur = root
    for part in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(part, {})
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            if not hasattr(cur, part):
                raise ValueError(f"optimization path does not exist: {path}")
            cur = getattr(cur, part)
    leaf = parts[-1]
    if isinstance(cur, dict):
        cur[leaf] = value
    elif isinstance(cur, list):
        cur[int(leaf)] = value
    else:
        if not hasattr(cur, leaf):
            raise ValueError(f"optimization path does not exist: {path}")
        setattr(cur, leaf, value)


def _parameter_key(param: OptimizationParameter) -> str:
    """Return the stable key shown in trial results."""
    stage = f".pipeline[{param.stage_index}]" if param.stage_index is not None else ""
    return f"{param.target}{stage}.{param.path}"


def _trial_config(config: PerfConfig, output_dir: Path) -> PerfConfig:
    """Prepare an isolated trial run that cannot perturb history baselines."""
    cfg = config.model_copy(deep=True)
    cfg.optimization.enabled = False
    cfg.report.output_dir = str(output_dir)
    cfg.report.history = False
    cfg.report.baseline = "none"
    cfg.report.regression.enabled = False
    if cfg.thresholds:
        cfg.thresholds.on_fail = "warn"
        cfg.thresholds.on_not_evaluated = "warn"
        cfg.thresholds.alerts = []
    return cfg


def _optimized_config(candidate: PerfConfig, base: PerfConfig) -> PerfConfig:
    """Return the winning config with report settings restored."""
    cfg = candidate.model_copy(deep=True)
    cfg.report.output_dir = base.report.output_dir
    cfg.report.history = base.report.history
    cfg.report.baseline = base.report.baseline
    cfg.report.regression = base.report.regression
    cfg.optimization.enabled = False
    return cfg


def _write_config(config: PerfConfig, path: Path) -> None:
    """Write an effective YAML config."""
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))


def _latest_report(output_dir: Path) -> Path | None:
    """Return the newest llm-perf JSON report in a trial directory."""
    reports = sorted(output_dir.glob("llm-perf-*.json"), key=lambda p: p.stat().st_mtime)
    return reports[-1] if reports else None


def _run_cli(config_path: Path) -> int:
    """Run one candidate through the normal llm-perf CLI in a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "llm_perf.cli",
        "run",
        "-c",
        str(config_path),
        "--headless",
    ]
    proc = subprocess.run(cmd, check=False)  # noqa: S603 - fixed interpreter/module invocation
    return proc.returncode


def _constraint_violations(
    constraints: ThresholdRule,
    scenario: dict,
    report: dict,
    scenario_name: str,
) -> list[str]:
    """Evaluate optimization constraints against an aggregate report."""
    violations: list[str] = []
    for rule_name, metric, comparator in _CONSTRAINT_FIELDS:
        limit = getattr(constraints, rule_name, None)
        if limit is None:
            continue
        actual = scenario.get(metric)
        if actual is None or not _compare(actual, comparator, limit):
            violations.append(f"constraint {rule_name}: actual={actual} {comparator} {limit} not satisfied")

    stage_rows = [s for s in report.get("stages", []) if s.get("scenario") in (scenario_name, None)]
    for stage_name, rules in (constraints.stages or {}).items():
        row = next((s for s in stage_rows if s.get("stage") == stage_name), None)
        for metric, limit in rules.items():
            actual = row.get(metric) if row else None
            if actual is None or not _compare(actual, "<=", limit):
                violations.append(
                    f"constraint stages.{stage_name}.{metric}: actual={actual} <= {limit} not satisfied"
                )
    return violations


def _compare(actual: float, comparator: str, limit: float) -> bool:
    """Compare a metric against a constraint limit."""
    if comparator == "<=":
        return actual <= limit
    return actual >= limit
