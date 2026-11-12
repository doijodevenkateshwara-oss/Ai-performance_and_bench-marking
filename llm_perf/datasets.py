"""Dataset loading, sampling, and templating for prompts."""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from pathlib import Path

import yaml

from .config import DatasetConfig, DatasetSample


class Dataset:
    """Loaded + sampled dataset of prompt rows."""

    def __init__(self, name: str, rows: list[dict], config: DatasetConfig):
        self.name = name
        self.rows = rows
        self.config = config
        self.vars = dict(config.vars)

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, name: str, cfg: DatasetConfig) -> Dataset:
        fmt = cfg.format or _detect_format(cfg.source)
        rows = _read_rows(cfg.source, fmt)
        rows = [_map_columns(r, cfg.columns) for r in rows]
        rows = _apply_sample(rows, cfg.sample)
        return cls(name, rows, cfg)

    # -- iteration -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def sample(self, rng: random.Random) -> dict:
        """Pick a row and render {var} placeholders + vars pools."""
        if not self.rows:
            return {}
        row = rng.choice(self.rows) if len(self.rows) > 1 else self.rows[0]
        return self.render(row, rng)

    def render(self, row: dict, rng: random.Random) -> dict:
        """Fill {var} placeholders in string fields using row fields then vars pools."""
        out = dict(row)
        for key, val in list(out.items()):
            if isinstance(val, str) and "{" in val:
                out[key] = _render_template(val, row, self.vars, rng)
        # fill vars pools into a special 'vars' dict for prompt templates
        out["vars"] = {k: rng.choice(v) if isinstance(v, list) and v else v for k, v in self.vars.items()}
        return out

    def shard(self, worker_index: int, num_workers: int) -> Dataset:
        """Deterministic shard for distributed runs."""
        if num_workers <= 1:
            return self
        rows = [r for i, r in enumerate(self.rows) if i % num_workers == worker_index]
        return Dataset(self.name, rows, self.config)


def _detect_format(source: str) -> str:
    s = source.split("?", maxsplit=1)[0].lower()
    if s.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if s.endswith(".csv"):
        return "csv"
    if s.endswith((".yaml", ".yml")):
        return "yaml"
    if s.endswith(".json"):
        # treat .json as jsonl if multiple objects, else yaml-safe; default jsonl-ish
        return "jsonl"
    return "yaml"


def _read_rows(source: str, fmt: str) -> list[dict]:
    text = _read_text(source)
    if fmt == "yaml":
        data = yaml.safe_load(text) or []
        if isinstance(data, dict):
            for key in ("rows", "data", "items", "prompts"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return list(data)
    if fmt == "csv":
        return [dict(r) for r in csv.DictReader(io.StringIO(text))]
    # jsonl
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_text(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as r:
            return r.read().decode("utf-8")
    if source.startswith("s3://"):
        import boto3

        bucket, key = source[5:].split("/", 1)
        s3 = boto3.client("s3")
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    if source.startswith("gs://"):
        import gcsfs

        with gcsfs.GCSFileSystem().open(source[5:], "rb") as f:
            return f.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def _map_columns(row: dict, columns: dict[str, str]) -> dict:
    """Rename source fields to logical names. `columns` maps logical->source."""
    if not columns:
        return row
    out = dict(row)
    for logical, source_field in columns.items():
        if source_field in row:
            out[logical] = row[source_field]
    return out


def _apply_sample(rows: list[dict], sample: DatasetSample | None) -> list[dict]:
    if not sample:
        return rows
    if sample.strategy == "first" and sample.count:
        return rows[: sample.count]
    rng = random.Random(sample.seed)
    pool = list(rows)
    if sample.strategy == "random":
        rng.shuffle(pool)
        if sample.count:
            pool = pool[: sample.count]
    elif sample.strategy == "weighted" and sample.weights:
        # weight by a row field: weights maps field-value -> weight
        next(iter(sample.weights)) if sample.weights else None
        # allow weights keyed by column name with nested dict, else treat keys as values of 'category'/'tag'/'label'
        field = _guess_weight_field(rows, sample.weights)
        weighted = []
        for r in pool:
            w = sample.weights.get(str(r.get(field, "")), 1.0)
            weighted.extend([r] * max(1, int(w * 10)))
        rng.shuffle(weighted)
        pool = weighted
        if sample.count:
            pool = pool[: sample.count]
    return pool


def _guess_weight_field(rows: list[dict], weights: dict) -> str:
    keys = set(weights)
    for field in ("category", "tag", "label", "type", "class"):
        if any(str(r.get(field, "")) in keys for r in rows[:50]):
            return field
    # fall back: find any field whose values intersect weight keys
    if rows:
        for f in rows[0]:
            if any(str(r.get(f, "")) in keys for r in rows[:50]):
                return f
    return "category"


def _render_template(template: str, row: dict, vars_: dict, rng: random.Random) -> str:
    import re

    def repl(m):
        key = m.group(1).strip()
        if key in row:
            return str(row[key])
        if key in vars_:
            v = vars_[key]
            return str(rng.choice(v) if isinstance(v, list) and v else v)
        if key in row.get("vars", {}):
            return str(row["vars"][key])
        return m.group(0)

    return re.sub(r"\{\{?\s*([^{}]+?)\s*\}?\}", repl, template)


# ---------------------------------------------------------------------------
# Trace import
# ---------------------------------------------------------------------------


def import_traces(path: str, fmt: str = "jsonl", mapping: dict | None = None) -> list[dict]:
    """Convert production trace exports into dataset rows.

    Supported fmts: 'jsonl' (generic), 'langsmith', 'langfuse'.
    mapping: {logical_field: jmespath-ish dotted path} e.g. {'prompt': 'inputs.question'}
    """
    rows = _read_rows(path, "jsonl")
    out = []
    for r in rows:
        if fmt == "langsmith":
            out.append(
                {
                    "prompt": _dig(r, "inputs.question") or _dig(r, "inputs.input") or "",
                    "expected": _dig(r, "outputs.answer"),
                    "raw": r,
                }
            )
        elif fmt == "langfuse":
            out.append(
                {
                    "prompt": _dig(r, "input") or _dig(r, "input.prompt") or "",
                    "expected": _dig(r, "output"),
                    "raw": r,
                }
            )
        elif mapping:
            out.append({k: _dig(r, v) for k, v in mapping.items()})
        else:
            out.append(r)
    return [r for r in out if r.get("prompt")]


def _dig(obj, path: str):
    cur = obj
    for part in str(path).split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            return None
    return cur


def write_dataset(rows: list[dict], dest: str, fmt: str = "jsonl") -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        Path(dest).write_text("\n".join(json.dumps(r) for r in rows))
    elif fmt == "yaml":
        Path(dest).write_text(yaml.safe_dump(rows, sort_keys=False))
    elif fmt == "csv":
        if not rows:
            Path(dest).write_text("")
            return
        keys = list(rows[0].keys())
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
        Path(dest).write_text(buf.getvalue())
