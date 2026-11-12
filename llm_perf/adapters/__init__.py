"""Framework adapters for user apps (LangChain / LlamaIndex / generic).

The actual layer implementation lives in layers/app_object.py; this module
holds optional framework-specific helpers (result unwrapping, tracing hooks).


Author: Venkateshwara Doijode
"""

from __future__ import annotations


def langchain_result_text(result) -> str:
    """Unwrap a LangChain result to plain text via the shared extractor."""
    from ..layers.app_object import _extract_text

    return _extract_text(result)


def llamaindex_result_text(result) -> str:
    """Unwrap a LlamaIndex result to plain text via the shared extractor."""
    from ..layers.app_object import _extract_text

    return _extract_text(result)
