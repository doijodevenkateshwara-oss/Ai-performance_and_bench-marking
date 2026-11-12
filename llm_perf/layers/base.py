"""Layer protocol — every target type implements run()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..providers.base import NormalizedResponse


@dataclass
class LayerResult:
    output_text: str = ""
    response: NormalizedResponse | None = None
    stage_metrics: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    retrieved_texts: list[str] = field(default_factory=list)
    row: dict | None = None
    raw: Any = None
    error: str | None = None


class Layer(Protocol):
    def run(self, prompt: str, row: dict | None, scenario_params: dict) -> LayerResult: ...
