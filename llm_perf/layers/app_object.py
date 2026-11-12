"""Layer 2: app_object — import user's LangChain/LlamaIndex/generic object."""

from __future__ import annotations

import importlib
import inspect
import time

from ..config import AppObjectTarget
from ..metrics import record_error, request_span
from ..prompts import PromptPicker
from .base import LayerResult


def _import_factory(python_path: str):
    """Import 'module.path:callable' and return the callable."""
    mod_path, _, attr = python_path.partition(":")
    mod = importlib.import_module(mod_path)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class AppObjectLayer:
    def __init__(self, target: AppObjectTarget, picker: PromptPicker):
        self.target = target
        self.picker = picker
        self._app = None

    def _get_app(self):
        if self._app is None:
            factory = _import_factory(self.target.python_path)
            if inspect.iscoroutinefunction(factory) or not inspect.isfunction(factory):
                app = factory
            else:
                app = factory(**self.target.init_args)
            self._app = app
        return self._app

    async def _acall(self, app, method: str, prompt: str, row: dict | None):
        fn = getattr(app, method, None)
        if fn is None:
            raise AttributeError(f"app has no method {method!r}")
        # llamaindex query engines take positional query
        if self.target.framework == "llamaindex" or method in ("query", "aquery"):
            result = fn(prompt)
        else:
            result = fn(input=prompt) if "input" in inspect.signature(fn).parameters else fn(prompt)
        if inspect.isawaitable(result):
            result = await result
        return result

    def run(
        self, prompt: str, row: dict | None, scenario_params: dict, scenario_name: str = "default"
    ) -> LayerResult:
        # app_object calls are async-friendly; run via asyncio in the greenlet
        import asyncio

        t = self.target
        with request_span(scenario_name, t.name, f"app_{t.framework}") as span:
            start = time.monotonic()
            try:
                app = self._get_app()
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(self._acall(app, t.call, prompt, row))
                finally:
                    loop.close()
                e2e = time.monotonic() - start
                text = _extract_text(result)
                from ..metrics import instruments, log_request_event

                instruments()["request_total"].add(
                    1, {"scenario": scenario_name, "provider": f"app_{t.framework}", "result": "success"}
                )
                instruments()["request_duration"].record(
                    e2e, {"scenario": scenario_name, "provider": f"app_{t.framework}"}
                )
                log_request_event(scenario_name, e2e * 1000)
                span.set_attribute("llm_perf.output_chars", len(text))
                return LayerResult(output_text=text, row=row, stage_metrics={"e2e_s": e2e}, raw=result)
            except Exception as e:
                e2e = time.monotonic() - start
                record_error(scenario_name, f"app_{t.framework}", str(e), e2e)
                from ..metrics import log_request_event

                log_request_event(scenario_name, e2e * 1000, exception=str(e))
                span.set_attribute("error", True)
                return LayerResult(error=str(e), row=row, stage_metrics={"e2e_s": e2e})


def _extract_text(result) -> str:
    """Best-effort text extraction from LC/LlamaIndex/generic results."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    # LangChain AIMessage / dict with content
    content = getattr(result, "content", None)
    if content is not None:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    if isinstance(result, dict):
        for k in ("output_text", "text", "content", "response", "answer"):
            if k in result:
                v = result[k]
                return v if isinstance(v, str) else str(v)
        if "generations" in result:
            try:
                return result["generations"][0][0]["text"]
            except Exception:
                pass
    return str(result)
