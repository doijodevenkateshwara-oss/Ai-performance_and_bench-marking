"""Tests for the app_object layer — async execution under gevent greenlets.

Regression coverage for the "event loop is already running" failure:
Locust greenlets share one OS thread, so concurrent async app calls must
each drive their own event loop cooperatively.

Author: Venkateshwara Doijode
"""

import asyncio

import gevent

from llm_perf.config import AppObjectTarget
from llm_perf.layers.app_object import AppObjectLayer
from llm_perf.prompts import PromptPicker


def _layer(call: str = "ainvoke") -> AppObjectLayer:
    target = AppObjectTarget(name="t", framework="generic", python_path="x:y", call=call)
    return AppObjectLayer(target, PromptPicker())


class _AsyncApp:
    """App whose method genuinely awaits — forces loop interleaving."""

    async def ainvoke(self, input: str = "") -> str:
        await asyncio.sleep(0.01)
        return f"ok:{input}"


class _SyncApp:
    def invoke(self, input: str = "") -> str:
        return f"sync:{input}"


class TestAppObjectAsync:
    def test_concurrent_async_calls_no_loop_errors(self):
        """8 greenlets awaiting asyncio.sleep concurrently must all succeed."""
        layer = _layer()
        layer._app = _AsyncApp()
        errors = []

        def work(i):
            r = layer.run(f"p{i}", None, {}, "s")
            errors.append(r.error)

        jobs = [gevent.spawn(work, i) for i in range(8)]
        gevent.joinall(jobs, timeout=10)
        assert all(j.ready() for j in jobs), "greenlets did not finish"
        assert errors == [None] * 8

    def test_sync_call_still_works(self):
        layer = _layer(call="invoke")
        layer._app = _SyncApp()
        r = layer.run("hello", None, {}, "s")
        assert r.error is None
        assert r.output_text == "sync:hello"

    def test_async_result_text_extracted(self):
        layer = _layer()
        layer._app = _AsyncApp()
        r = layer.run("q", None, {}, "s")
        assert r.error is None
        assert r.output_text == "ok:q"

    def test_app_exception_is_captured(self):
        class _Boom:
            async def ainvoke(self, input: str = ""):
                raise ValueError("boom")

        layer = _layer()
        layer._app = _Boom()
        r = layer.run("x", None, {}, "s")
        assert r.error is not None
        assert "boom" in r.error
