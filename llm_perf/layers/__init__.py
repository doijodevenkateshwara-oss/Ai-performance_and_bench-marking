"""Layer dispatch — builds the right Layer for a target."""
from __future__ import annotations

from typing import Any, Optional

from ..config import (AppEndpointTarget, AppObjectTarget, LlmApiTarget,
                      PipelineTarget, Target)
from ..datasets import Dataset
from ..prompts import PromptPicker, load_prompts
from .app_endpoint import AppEndpointLayer
from .app_object import AppObjectLayer
from .base import Layer, LayerResult
from .llm_api import LlmApiLayer
from .pipeline import PipelineLayer


class LayerFactory:
    """Caches layers + datasets + prompt pickers per worker process."""

    def __init__(self, config):
        self.config = config
        self._layers: dict[str, Layer] = {}
        self._datasets: dict[str, Dataset] = {}

    def dataset(self, name: str) -> Optional[Dataset]:
        if name in self._datasets:
            return self._datasets[name]
        cfg = self.config.datasets.get(name)
        if cfg is None:
            return None
        ds = Dataset.load(name, cfg)
        # deterministic shard for distributed workers
        try:
            from locust.env import get_environment
            runner = get_environment().runner
            if runner and getattr(runner, "worker_index", None) is not None \
               and getattr(runner, "worker_count", 1):
                ds = ds.shard(runner.worker_index, runner.worker_count)
        except Exception:
            pass
        self._datasets[name] = ds
        return ds

    def layer(self, target_name: str) -> Layer:
        if target_name in self._layers:
            return self._layers[target_name]
        target = next(t for t in self.config.targets if t.name == target_name)
        prompts = load_prompts(getattr(target, "prompts", None))
        ds = self.dataset(getattr(target, "dataset", None) or "") if getattr(target, "dataset", None) else None
        picker = PromptPicker(prompts=prompts, dataset=ds,
                              vars_=getattr(target, "vars", None))
        if isinstance(target, LlmApiTarget):
            layer = LlmApiLayer(target, picker)
        elif isinstance(target, PipelineTarget):
            layer = PipelineLayer(target, picker)
        elif isinstance(target, AppEndpointTarget):
            layer = AppEndpointLayer(target, picker)
        elif isinstance(target, AppObjectTarget):
            layer = AppObjectLayer(target, picker)
        else:
            raise ValueError(f"unsupported target type: {type(target)}")
        self._layers[target_name] = layer
        return layer

    def prompt_and_row(self, layer: Layer, rng) -> tuple[str, Optional[dict]]:
        """Pick prompt+dataset row for layers that support datasets."""
        picker = getattr(layer, "picker", None)
        ds = getattr(picker, "dataset", None) if picker else None
        if ds is not None and len(ds.rows):
            row = ds.sample(rng)
            prompt = ""
            for key in ("prompt", "question", "input", "text"):
                if key in row and row[key]:
                    prompt = str(row[key])
                    break
            if not prompt:
                prompt = picker.pick(rng, row)
            return prompt, row
        return picker.pick(rng) if picker else "", None
