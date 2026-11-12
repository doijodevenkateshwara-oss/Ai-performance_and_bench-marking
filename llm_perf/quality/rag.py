"""RAG-specific quality: context recall against dataset expected_context."""
from __future__ import annotations

import re
from typing import Optional


def context_recall(retrieved_texts: list[str], expected_context: Optional[str]) -> Optional[float]:
    """Fraction of expected_context sentences/keywords found in retrieved context.

    Simple token-overlap recall: splits expected context into sentences, checks
    what fraction appear (fuzzy substring) in the concatenated retrieved text.
    """
    if not expected_context:
        return None
    retrieved = " ".join(retrieved_texts).lower()
    # split expected into sentences or lines
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", expected_context) if len(p.strip()) > 15]
    if not parts:
        parts = [expected_context.strip()]
    hits = 0
    for p in parts:
        # check key content words overlap (>=60% of significant words present)
        words = [w for w in re.findall(r"[a-z0-9]{4,}", p.lower())]
        if not words:
            continue
        present = sum(1 for w in set(words) if w in retrieved)
        if present / len(set(words)) >= 0.6:
            hits += 1
    return hits / len(parts) if parts else None
