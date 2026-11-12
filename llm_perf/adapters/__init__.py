"""Framework adapters for user apps (LangChain / LlamaIndex / generic).

The actual layer implementation lives in layers/app_object.py; this module
holds optional framework-specific helpers (result unwrapping, tracing hooks).
"""
from __future__ import annotations


def langchain_result_text(result) -> str:
    from ..layers.app_object import _extract_text
    return _extract_text(result)


def llamaindex_result_text(result) -> str:
    from ..layers.app_object import _extract_text
    return _extract_text(result)
