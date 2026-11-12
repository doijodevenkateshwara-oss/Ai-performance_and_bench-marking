"""Workload profiles — translate load config into a schedule of phases.

A Phase is (target_users, spawn_rate, duration_s, label). Locust's
LoadTestShape returns the current phase's target/spawn_rate each tick and
handles the actual ramping, so phases only describe *levels to hold*.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import LoadConfig


@dataclass
class Phase:
    target_users: int
    spawn_rate: float
    duration_s: float
    label: str


def _seconds(value: str | None, default: float) -> float:
    """Parse a duration string ('90s', '2m', '1h', '1h30m', '90') or return default."""
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        pass
    total = 0.0
    import re

    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([smhd]?)", s, flags=re.I):
        mult = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit.lower()]
        total += float(num) * mult
    return total if total > 0 else default


def _spawn(load: LoadConfig) -> float:
    return float(load.spawn_rate or 1.0)


def _simple(
    load: LoadConfig, steady_s: float, ramp_up_s: float, ramp_down_s: float, label: str = "steady"
) -> list[Phase]:
    """ramp -> steady hold -> ramp down, shared by smoke/baseline/load/soak."""
    users = max(int(load.users), 1)
    sr = _spawn(load)
    phases = []
    if ramp_up_s > 0:
        phases.append(Phase(users, sr, ramp_up_s, "ramp_up"))
    if steady_s > 0:
        phases.append(Phase(users, sr, steady_s, label))
    if ramp_down_s > 0:
        phases.append(Phase(0, sr, ramp_down_s, "ramp_down"))
    return phases


def _step_phases(
    load: LoadConfig, levels: list[int], hold_s: float, final_hold_s: float | None = None
) -> list[Phase]:
    """Hold each concurrency level for hold_s; locust ramps between levels at spawn_rate."""
    sr = _spawn(load)
    phases = [Phase(u, sr, hold_s, f"step_{u}u") for u in levels]
    if final_hold_s and final_hold_s > 0:
        phases.append(Phase(levels[-1], sr, final_hold_s, "steady"))
    if levels[-1] > 0:
        phases.append(Phase(0, sr, max(hold_s / 4.0, 10.0), "ramp_down"))
    return phases


def build_phases(load: LoadConfig) -> list[Phase]:
    """Build the phase schedule for the configured profile (or custom stages)."""
    users = max(int(load.users), 1)
    sr = _spawn(load)
    profile = load.profile or ("custom" if load.stages else None)
    steady_s = _seconds(load.duration, 600.0)

    if profile == "custom" or (load.stages and not load.profile):
        phases = [
            Phase(
                max(int(s.users), 0), float(s.spawn_rate or sr), _seconds(s.duration, steady_s), f"stage_{i}"
            )
            for i, s in enumerate(load.stages or [])
        ]
        phases.append(Phase(0, sr, _seconds(load.ramp_down, 30.0), "ramp_down"))
        return phases

    if profile == "smoke":
        return _simple(
            load, _seconds(load.duration, 120.0), _seconds(load.ramp_up, 30.0), _seconds(load.ramp_down, 15.0)
        )

    if profile == "baseline":
        return _simple(
            load,
            _seconds(load.duration, 600.0),
            _seconds(load.ramp_up, 120.0),
            _seconds(load.ramp_down, 60.0),
        )

    if profile == "soak":
        return _simple(
            load,
            _seconds(load.duration, 4 * 3600.0),
            _seconds(load.ramp_up, 300.0),
            _seconds(load.ramp_down, 300.0),
        )

    if profile == "load":
        # default ramp = natural spawn time at configured spawn_rate
        ramp_up = _seconds(load.ramp_up, users / max(sr, 0.01))
        ramp_down = _seconds(load.ramp_down, ramp_up / 2.0)
        return _simple(load, steady_s, ramp_up, ramp_down)

    if profile == "stress":
        n = max(int(load.steps or 4), 1)
        step_s = _seconds(load.step_duration, steady_s)
        levels = sorted({max(1, round(users * i / n)) for i in range(1, n + 1)})
        return _step_phases(load, levels, step_s)

    if profile in ("scalability", "capacity"):
        step_users = max(int(load.step_users or max(1, users // 5)), 1)
        step_s = _seconds(load.step_duration, steady_s)
        levels = list(range(step_users, users + 1, step_users))
        if not levels or levels[-1] != users:
            levels.append(users)
        levels = sorted(set(levels))
        final = steady_s if profile == "scalability" else None
        return _step_phases(load, levels, step_s, final_hold_s=final)

    if profile == "spike":
        baseline = max(int(load.baseline_users or max(1, users // 10)), 1)
        phases = [
            Phase(users, sr, _seconds(load.ramp_up, 30.0), "spike_ramp"),
            Phase(users, sr, steady_s, "spike_hold"),
            Phase(baseline, sr, _seconds(load.baseline_duration, 120.0), "baseline_hold"),
            Phase(0, sr, _seconds(load.ramp_down, 30.0), "ramp_down"),
        ]
        return phases

    # no profile and no stages -> flat load handled by locust args (no shape)
    return []


def schedule_summary(load: LoadConfig) -> list[dict]:
    """JSON-serializable phase schedule for reports."""
    return [
        {
            "label": p.label,
            "target_users": p.target_users,
            "spawn_rate": p.spawn_rate,
            "duration_s": p.duration_s,
        }
        for p in build_phases(load)
    ]
