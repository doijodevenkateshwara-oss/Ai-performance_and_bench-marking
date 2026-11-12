"""llm-perf CLI."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from .config import PerfConfig

app = typer.Typer(help="Multi-layer LLM/agent performance testing framework")
dataset_app = typer.Typer(help="Dataset management commands")
app.add_typer(dataset_app, name="dataset")
console = Console()


def _load_or_exit(path: str, skip_connectivity: bool = False) -> PerfConfig:
    try:
        cfg = PerfConfig.from_yaml(path)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(2)
    if not skip_connectivity:
        _check_env_vars(cfg)
    return cfg


def _filter_config(cfg: PerfConfig, layers: str | None, scenarios: str | None) -> None:
    """Narrow targets/scenarios in place so layers can run independently."""
    if layers:
        keep = {x.strip() for x in layers.split(",") if x.strip()}
        cfg.targets = [t for t in cfg.targets if t.layer in keep]
        names = {t.name for t in cfg.targets}
        cfg.scenarios = [s for s in cfg.scenarios if s.target in names]
    if scenarios:
        keep_s = {x.strip() for x in scenarios.split(",") if x.strip()}
        cfg.scenarios = [s for s in cfg.scenarios if s.name in keep_s]
    if not cfg.targets or not cfg.scenarios:
        console.print("[red]Filter removed all targets/scenarios[/red]")
        raise typer.Exit(2)


def _check_env_vars(cfg: PerfConfig) -> None:
    missing = []
    for t in cfg.targets:
        env = getattr(t, "api_key_env", None)
        if env and not os.environ.get(env):
            missing.append(f"target {t.name}: ${env}")
    for name, ds in cfg.datasets.items():
        src = ds.source
        if not (src.startswith(("http", "s3://", "gs://")) or Path(src).exists()):
            missing.append(f"dataset {name}: source not found: {src}")
    if cfg.thresholds:
        for a in cfg.thresholds.alerts:
            env = a.get("url_env")
            if env and not os.environ.get(env):
                missing.append(f"alert: ${env}")
    if missing:
        console.print("[yellow]Warnings:[/yellow]")
        for m in missing:
            console.print(f"  - {m}")


@app.command()
def validate(
    config: str = typer.Option(..., "-c", "--config"),
    skip_connectivity: bool = typer.Option(False, "--skip-connectivity"),
):
    """Validate a config file (schema + env vars + dataset sources)."""
    cfg = _load_or_exit(config, skip_connectivity)
    console.print(
        f"[green]Config OK[/green]: {len(cfg.targets)} targets, "
        f"{len(cfg.scenarios)} scenarios, {len(cfg.datasets)} datasets"
    )
    for t in cfg.targets:
        console.print(f"  target: {t.name} (layer={t.layer})")
    for s in cfg.scenarios:
        console.print(f"  scenario: {s.name} -> {s.target} (weight={s.weight})")


@app.command()
def run(
    config: str = typer.Option(..., "-c", "--config"),
    users: int | None = typer.Option(None, "--users"),
    spawn_rate: float | None = typer.Option(None, "--spawn-rate"),
    duration: str | None = typer.Option(None, "--duration", "-t"),
    headless: bool = typer.Option(False, "--headless"),
    web_port: int = typer.Option(8089, "--web-port"),
    master: str | None = typer.Option(None, "--master"),
    workers: int | None = typer.Option(None, "--workers"),
    no_alerts: bool = typer.Option(False, "--no-alerts"),
    skip_connectivity: bool = typer.Option(False, "--skip-connectivity"),
    layers: str | None = typer.Option(
        None, "--layers", help="comma-separated layers to run, e.g. llm_api,pipeline"
    ),
    scenarios: str | None = typer.Option(None, "--scenarios", help="comma-separated scenario names to run"),
):
    """Run a performance test."""
    cfg = _load_or_exit(config, skip_connectivity)
    _filter_config(cfg, layers, scenarios)
    if no_alerts and cfg.thresholds:
        cfg.thresholds.alerts = []
    if users is not None:
        cfg.load.users = users
    if spawn_rate is not None:
        cfg.load.spawn_rate = spawn_rate
    if duration is not None:
        cfg.load.duration = duration

    # materialize effective config for the locustfile subprocess/in-process run
    effective = Path(config).resolve().parent / ".llm_perf_effective.yaml"
    import yaml

    effective.write_text(yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True), sort_keys=False))
    os.environ["LLM_PERF_CONFIG"] = str(effective)

    from locust.main import main as locust_main

    args = ["-f", "llm_perf.locustfile"]
    if headless or master or workers:
        args.append("--headless")
    args += ["-u", str(cfg.load.users), "-r", str(cfg.load.spawn_rate)]
    if cfg.load.duration:
        args += ["-t", cfg.load.duration]
    if master:
        args += ["--master", "--master-bind-host", master]
    if workers:
        args += ["--worker"]
    console.print(
        f"[cyan]Starting llm-perf[/cyan] config={config} "
        f"users={cfg.load.users} spawn={cfg.load.spawn_rate} "
        f"duration={cfg.load.duration or 'until stopped'}"
    )
    try:
        locust_main(args)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
        # check threshold failure flag via env set in locustfile
        if os.environ.get("LLM_PERF_THRESHOLDS_FAILED") == "1":
            raise typer.Exit(1)
        raise typer.Exit(code or 0)


@app.command("proto-gen")
def proto_gen_cmd(proto: str = typer.Option(..., "-p", "--proto")):
    """Pre-compile a .proto file for gRPC targets."""
    from .proto import proto_gen

    try:
        out = proto_gen(proto)
        console.print(f"[green]Generated stubs in[/green] {out}")
    except Exception as e:
        console.print(f"[red]proto-gen failed:[/red] {e}")
        raise typer.Exit(1)


@app.command("quality-check")
def quality_check(
    config: str = typer.Option(..., "-c", "--config"),
    response_file: str = typer.Option(..., "--response-file"),
    question: str = typer.Option("", "--question"),
):
    """Offline: score a saved response against a config's quality rules."""
    cfg = _load_or_exit(config, skip_connectivity=True)
    response = Path(response_file).read_text()
    # use first scenario's quality config
    if not cfg.scenarios:
        console.print("[red]No scenarios in config[/red]")
        raise typer.Exit(2)
    sc = cfg.scenarios[0]
    if not sc.quality:
        console.print("[yellow]Scenario has no quality config[/yellow]")
    from .quality import evaluate_quality

    res = evaluate_quality(sc.quality, question=question, response=response)
    console.print(f"score={res.score} passed={res.passed} tags={res.failure_tags}")
    if res.judge_reason:
        console.print(f"judge: {res.judge_reason}")


# ----------------------------- dataset commands -----------------------------


@dataset_app.command("list")
def ds_list(config: str = typer.Option(..., "-c", "--config")):
    cfg = _load_or_exit(config, skip_connectivity=True)
    for name, ds in cfg.datasets.items():
        console.print(f"{name}: {ds.source} (format={ds.format or 'auto'})")


@dataset_app.command("show")
def ds_show(name: str, config: str = typer.Option(..., "-c", "--config")):
    cfg = _load_or_exit(config, skip_connectivity=True)
    ds_cfg = cfg.datasets.get(name)
    if not ds_cfg:
        console.print(f"[red]Unknown dataset {name}[/red]")
        raise typer.Exit(2)
    from .datasets import Dataset

    ds = Dataset.load(name, ds_cfg)
    console.print(f"{name}: {len(ds.rows)} rows")
    for r in ds.rows[:5]:
        console.print(" ", str(r)[:200])


@dataset_app.command("sample")
def ds_sample(name: str, config: str = typer.Option(..., "-c", "--config"), n: int = typer.Option(5, "-n")):
    import random

    cfg = _load_or_exit(config, skip_connectivity=True)
    ds_cfg = cfg.datasets.get(name)
    if not ds_cfg:
        raise typer.Exit(2)
    from .datasets import Dataset

    ds = Dataset.load(name, ds_cfg)
    rng = random.Random(0)
    for _ in range(min(n, len(ds.rows))):
        console.print(str(ds.sample(rng))[:300])


@dataset_app.command("import")
def ds_import(
    input: str = typer.Option(..., "--from"),
    dest: str = typer.Option(..., "--to"),
    fmt: str = typer.Option("jsonl", "--format"),
    trace_fmt: str = typer.Option("jsonl", "--trace-format"),
    mapping: str | None = typer.Option(None, "--map", help="logical=path pairs, comma separated"),
):
    """Import traces/rows into a dataset file."""
    from .datasets import import_traces, write_dataset

    map_dict = None
    if mapping:
        map_dict = dict(kv.split("=", 1) for kv in mapping.split(","))
    if trace_fmt in ("langsmith", "langfuse"):
        rows = import_traces(input, fmt=trace_fmt)
    else:
        rows = import_traces(input, fmt="jsonl", mapping=map_dict)
    write_dataset(rows, dest, fmt=fmt)
    console.print(f"[green]Wrote {len(rows)} rows to {dest}[/green]")


@dataset_app.command("validate")
def ds_validate(path: str):
    """Validate a standalone dataset file."""
    from .datasets import _detect_format, _read_rows

    try:
        fmt = _detect_format(path)
        rows = _read_rows(path, fmt)
        console.print(f"[green]OK[/green]: {len(rows)} rows ({fmt})")
        if rows:
            console.print("columns:", list(rows[0].keys()))
    except Exception as e:
        console.print(f"[red]Invalid dataset:[/red] {e}")
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()
