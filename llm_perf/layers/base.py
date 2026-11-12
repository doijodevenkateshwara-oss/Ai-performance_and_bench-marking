"""Layer protocol — every target type implements run()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..providers.base import NormalizedResponse


@dataclass
class LayerResult:
    output_text: str = ""
    response: Optional[NormalizedResponse] = None
    stage_metrics: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    retrieved_texts: list[str] = field(default_factory=list)
    row: Optional[dict] = None
    raw: Any = None
    error: Optional[str] = None


class Layer(Protocol):
    def run(self, prompt: str, row: Optional[dict], scenario_params: dict) -> LayerResult:
        ...
