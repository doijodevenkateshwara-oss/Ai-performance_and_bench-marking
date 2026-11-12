"""Zero-cost heuristic quality checks.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config import HeuristicChecks


@dataclass
class CheckResult:
    """One named check outcome (pass/fail + detail + weight)."""

    name: str
    passed: bool
    detail: str = ""
    weight: float = 1.0


@dataclass
class HeuristicReport:
    """Collection of check results with aggregate score/tags."""

    checks: list[CheckResult] = field(default_factory=list)
    score: float = 1.0
    failure_tags: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True when every check passed (vacuously true with no checks)."""
        return all(c.passed for c in self.checks) if self.checks else True


def run_heuristics(text: str, cfg: HeuristicChecks, row: dict | None = None) -> HeuristicReport:
    """Run all configured zero-cost checks against a response."""
    checks: list[CheckResult] = []
    text = text or ""

    if cfg.non_empty:
        checks.append(CheckResult("non_empty", bool(text.strip()), "" if text.strip() else "empty response"))
    if cfg.min_chars:
        checks.append(CheckResult("min_chars", len(text) >= cfg.min_chars, f"len={len(text)}"))
    if cfg.max_chars:
        checks.append(CheckResult("max_chars", len(text) <= cfg.max_chars, f"len={len(text)}"))
    for pat in cfg.required_patterns:
        ok = re.search(pat, text, re.IGNORECASE | re.DOTALL) is not None
        checks.append(CheckResult(f"required:{pat}", ok, "" if ok else f"pattern {pat!r} not found"))
    for pat in cfg.forbidden_patterns:
        ok = re.search(pat, text, re.IGNORECASE | re.DOTALL) is None
        checks.append(CheckResult(f"forbidden:{pat}", ok, "" if ok else f"pattern {pat!r} found"))
    for s in cfg.contains:
        ok = s.lower() in text.lower()
        checks.append(CheckResult(f"contains:{s}", ok, "" if ok else f"{s!r} not in response"))
    if cfg.json_valid:
        try:
            json.loads(text)
            checks.append(CheckResult("json_valid", True))
        except Exception:
            checks.append(CheckResult("json_valid", False, "response is not valid JSON"))
    if cfg.exact_match_field and row is not None:
        expected = row.get(cfg.exact_match_field)
        ok = expected is not None and text.strip() == str(expected).strip()
        checks.append(CheckResult("exact_match", ok, "" if ok else "response != expected"))
    # refusal detection (always-on light check)
    refusal = re.search(
        r"\b(i('m| am) sorry|i cannot|as an ai|unable to (help|assist)|" r"i can't help with that)\b",
        text,
        re.IGNORECASE,
    )
    if refusal:
        checks.append(CheckResult("no_refusal", False, f"refusal pattern: {refusal.group(0)!r}"))

    passed = [c for c in checks if c.passed]
    total_w = sum(c.weight for c in checks) or 1.0
    score = sum(c.weight for c in passed) / total_w
    tags = [c.name for c in checks if not c.passed]
    return HeuristicReport(checks=checks, score=round(score, 4), failure_tags=tags)
