"""LLM-as-judge quality scoring via any configured provider.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re

from ..config import JudgeConfig
from ..providers import RequestSpec, call_llm

JUDGE_PROMPT = """You are an impartial evaluator. {rubric}

[QUESTION]
{question}

[RESPONSE]
{response}

{context_block}Return ONLY a JSON object: {{"score": <number between {lo} and {hi}>, "reason": "<one sentence>"}}"""

CONTEXT_BLOCK = """[CONTEXT]
{context}

"""


def _build_judge_prompt(cfg: JudgeConfig, question: str, response: str, context: str | None = None) -> str:
    """Render the judge prompt with rubric/question/response/context."""
    lo, hi = cfg.scale
    ctx = CONTEXT_BLOCK.format(context=context[:4000]) if context else ""
    return JUDGE_PROMPT.format(
        rubric=cfg.rubric, question=question[:4000], response=response[:6000], context_block=ctx, lo=lo, hi=hi
    )


def _parse_score(text: str, lo: int, hi: int) -> tuple[float | None, str]:
    """Extract score from judge output; normalize to 0..1."""
    # try JSON first
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    candidate = None
    if m:
        with contextlib.suppress(Exception):
            candidate = json.loads(m.group(0)).get("score")
    if candidate is None:
        m2 = re.search(r"score[^\d\-]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if m2:
            candidate = float(m2.group(1))
    if candidate is None:
        return None, "unparseable judge output"
    try:
        val = float(candidate)
    except (TypeError, ValueError):
        return None, "unparseable judge output"
    if val > hi:  # model returned 0-100 style
        val = val / 100.0 * (hi - lo) + lo
    norm = (val - lo) / max(hi - lo, 1e-9)
    return max(0.0, min(1.0, norm)), "ok"


def judge_score(
    cfg: JudgeConfig, question: str, response: str, context: str | None = None, timeout_s: float = 60.0
) -> tuple[float | None, str]:
    """Returns (normalized_score 0..1, reason)."""
    prompt = _build_judge_prompt(cfg, question, response, context)
    params = dict(cfg.params)
    params.setdefault("model", cfg.model or "gpt-4o")
    params.setdefault("max_tokens", 256)
    params.setdefault("temperature", 0.0)
    spec = RequestSpec(
        provider=cfg.provider,
        prompt=prompt,
        params=params,
        stream=False,
        base_url=cfg.base_url,
        api_key=os.environ.get(cfg.api_key_env) if cfg.api_key_env else None,
        timeout_s=timeout_s,
    )
    try:
        resp = call_llm(spec)
    except Exception as e:
        return None, f"judge call failed: {e}"
    text = resp.text or ""
    if not text:
        return None, "empty judge response"
    score, reason = _parse_score(text, *cfg.scale)
    return score, reason


class JudgeSampler:
    """Decides whether to run the judge on a given response."""

    def __init__(self, sample_rate: float, seed: int = 1234):
        """JudgeSampler(sample_rate, seed): deterministic sampling RNG."""
        self.sample_rate = sample_rate
        self._rng = random.Random(seed)

    def should_judge(self) -> bool:
        """True when this response should be sent to the LLM judge."""
        return self._rng.random() < self.sample_rate
