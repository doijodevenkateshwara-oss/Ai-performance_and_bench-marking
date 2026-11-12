"""Layer 2: app_object — import user's LangChain/LlamaIndex/generic object.

Author: Venkateshwara Doijode
"""

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
    from ..security import python_execution_allowed

    if not python_execution_allowed(f"app_object import {python_path}"):
        raise PermissionError(
            "python execution disabled by security.python_execution=off " f"(python_path={python_path!r})"
        )
    mod_path, _, attr = python_path.partition(":")
    mod = importlib.import_module(mod_path)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class AppObjectLayer:
    """Runs an in-process LangChain/LlamaIndex/generic Python object."""

    def __init__(self, target: AppObjectTarget, picker: PromptPicker):
        """AppObjectLayer(target, picker): lazily imports the app on first use."""
        self.target = target
        self.picker = picker
        self._app = None
        self._loop = None

    def _get_app(self):
        """Import python_path and instantiate the app once per worker."""
        if self._app is None:
            factory = _import_factory(self.target.python_path)
            if inspect.iscoroutinefunction(factory) or not inspect.isfunction(factory):
                app = factory
            else:
                app = factory(**self.target.init_args)
            self._app = app
        return self._app

    async def _acall(self, app, method: str, prompt: str, row: dict | None):
        """Invoke the configured method, awaiting coroutines as needed."""
        fn = getattr(app, method, None)
        if fn is None:
            raise AttributeError(f"app has no method {method!r}")
        # llamaindex query engines take positional query
        if self.target.framework == "llamaindex" or method in ("query", "aquery"):
            result = fn(prompt)
        else:
            try:
                takes_input = "input" in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                takes_input = False  # builtins/C-callables: fall back to positional
            result = fn(input=prompt) if takes_input else fn(prompt)
        if inspect.isawaitable(result):
            result = await result
        return result

    def run(
        self, prompt: str, row: dict | None, scenario_params: dict, scenario_name: str = "default"
    ) -> LayerResult:
        # app_object calls are async-friendly; run via asyncio in the greenlet
        """Execute the app call on a private event loop and record metrics."""
        import asyncio

        t = self.target
        with request_span(scenario_name, t.name, f"app_{t.framework}") as span:
            start = time.monotonic()
            try:
                app = self._get_app()
                # reuse one loop per layer — creating/closing per request is wasteful
                if self._loop is None or self._loop.is_closed():
                    self._loop = asyncio.new_event_loop()
                result = self._loop.run_until_complete(self._acall(app, t.call, prompt, row))
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
                return LayerResult(output_text=text, row=row, raw=result)
            except Exception as e:
                e2e = time.monotonic() - start
                record_error(scenario_name, f"app_{t.framework}", str(e), e2e)
                from ..metrics import log_request_event

                log_request_event(scenario_name, e2e * 1000, exception=str(e))
                span.set_attribute("error", True)
                return LayerResult(
                    error=str(e),
                    row=row,
                )


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
