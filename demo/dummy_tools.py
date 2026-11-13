"""Dummy function tools for the agent-stage demo.

Author: Venkateshwara Doijode
"""

from __future__ import annotations


def echo(input: str = "", **kwargs) -> dict:
    """Echo tool: returns its input wrapped in a result dict."""
    return {"echo": input or next(iter(kwargs.values()), "")}


def word_count(input: str = "") -> dict:
    """Count whitespace-separated words in the input."""
    return {"words": len(str(input).split())}
