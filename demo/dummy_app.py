"""Dummy in-process app for the app_object layer demo (generic framework).

Author: Venkateshwara Doijode
"""

from __future__ import annotations


class DummyApp:
    """Minimal app object: echoes the prompt back with a prefix."""

    def __init__(self, prefix: str = "dummy-app"):
        """DummyApp(prefix): store the response prefix."""
        self.prefix = prefix

    def invoke(self, input: str = "") -> str:
        """Sync invoke: return a deterministic answer."""
        return f"{self.prefix}: {input}"

    async def ainvoke(self, input: str = "") -> str:
        """Async invoke variant used when call=ainvoke."""
        return self.invoke(input)


def build(**kwargs) -> DummyApp:
    """Factory referenced by app_object python_path 'demo.dummy_app:build'."""
    return DummyApp(**kwargs)
