"""Prompt pools and {var} templating.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


def load_prompts(spec: Any) -> list[str]:
    """Accept: list[str], path to yaml/csv/txt file, or inline string."""
    if spec is None:
        return []
    if isinstance(spec, str):
        p = Path(spec)
        if p.exists():
            return _load_file(p)
        return [spec]
    if isinstance(spec, list):
        return [str(x) for x in spec]
    return []


def _load_file(path: Path) -> list[str]:
    """Load prompts from a yaml/csv/txt file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text) or []
        if isinstance(data, dict):
            for k in ("prompts", "items", "rows"):
                if k in data and isinstance(data[k], list):
                    return [str(x.get("prompt", x) if isinstance(x, dict) else x) for x in data[k]]
            return [str(v) for v in data.values()]
        return [str(x.get("prompt", x) if isinstance(x, dict) else x) for x in data]
    if path.suffix == ".csv":
        import csv
        import io

        return [r.get("prompt", "") for r in csv.DictReader(io.StringIO(text)) if r.get("prompt")]
    # txt: one prompt per line (blank-line separated blocks kept whole)
    blocks, cur = [], []
    for line in text.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


class PromptPicker:
    """Picks prompts from inline list or dataset rows, renders {var} placeholders."""

    def __init__(
        self, prompts: list[str] | None = None, dataset: Any | None = None, vars_: dict | None = None
    ):
        """PromptPicker(prompts, dataset, vars_)."""
        self.prompts = prompts or []
        self.dataset = dataset
        self.vars = vars_ or {}

    def _render(self, text: str, row: dict | None, rng: random.Random) -> str:
        """Substitute {var} placeholders using row fields then vars pools."""
        import re

        def repl(m):
            """Regex replacer for a single {var} placeholder."""
            key = m.group(1).strip().strip("{}")
            if row and key in row:
                return str(row[key])
            if key in self.vars:
                v = self.vars[key]
                return str(rng.choice(v) if isinstance(v, list) and v else v)
            return m.group(0)

        return re.sub(r"\{\{?\s*([A-Za-z_][\w.]*)\s*\}?\}", repl, text)

    def pick(self, rng: random.Random, row: dict | None = None) -> str:
        """Choose a prompt: row keys -> dataset sample -> prompt pool; render vars."""
        text = ""
        if row is not None:
            for key in ("prompt", "question", "input", "text"):
                if row.get(key):
                    text = str(row[key])
                    break
        # only sample the dataset when no row was supplied — sampling a second
        # row here would mismatch the prompt against the caller's row context
        if not text and row is None and self.dataset is not None:
            sampled = self.dataset.sample(rng)
            for key in ("prompt", "question", "input", "text"):
                if sampled.get(key):
                    text = str(sampled[key])
                    break
            if not text and sampled:
                text = str(next(iter(sampled.values())))
        if not text and self.prompts:
            text = rng.choice(self.prompts)
        return self._render(text, row, rng) if text else ""
