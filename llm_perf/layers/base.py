"""Layer protocol — every target type implements run().

Author: Venkateshwara Doijode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..providers.base import NormalizedResponse


@dataclass
class LayerResult:
    """Uniform result returned by every layer's run()."""

    output_text: str = ""
    response: NormalizedResponse | None = None
    retrieved_texts: list[str] = field(default_factory=list)
    row: dict | None = None
    raw: Any = None
    error: str | None = None
    # True when the layer already recorded per-stage cost breakdowns
    # (pipeline engine does); the task layer must not re-attribute the
    # aggregate response cost or cost_breakdown would double-count.
    cost_attributed: bool = False


class Layer(Protocol):
    """Protocol every target layer implements: run(prompt, row, params)."""

    def run(self, prompt: str, row: dict | None, scenario_params: dict) -> LayerResult:
        """Execute one request; return a LayerResult."""
        ...
