"""Combine heuristic + judge + rag signals into a final quality score."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import QualityConfig
from .heuristics import HeuristicReport, run_heuristics
from .judge import JudgeSampler, judge_score


@dataclass
class QualityResult:
    score: float | None = None  # normalized 0..1
    passed: bool | None = None
    heuristic: HeuristicReport | None = None
    judge_score: float | None = None
    judge_reason: str = ""
    context_recall: float | None = None
    failure_tags: list[str] = field(default_factory=list)
    judged: bool = False


def evaluate_quality(
    cfg: QualityConfig | None,
    *,
    question: str,
    response: str,
    row: dict | None = None,
    retrieved_texts: list[str] | None = None,
    expected_context: str | None = None,
    judge_sampler: JudgeSampler | None = None,
) -> QualityResult:
    if cfg is None:
        return QualityResult()
    res = QualityResult()

    # 1. heuristics (always, free)
    h = run_heuristics(response, cfg.heuristics, row=row)
    res.heuristic = h
    res.failure_tags.extend(h.failure_tags)

    # 2. rag context recall (free, if expected context provided)
    if cfg.rag and expected_context is not None and retrieved_texts is not None:
        from .rag import context_recall

        res.context_recall = context_recall(retrieved_texts, expected_context)

    # 3. LLM-as-judge (sampled)
    if cfg.judge is not None:
        sampler = judge_sampler or JudgeSampler(cfg.sample_rate)
        if sampler.should_judge():
            res.judged = True
            s, reason = judge_score(
                cfg.judge,
                question=question,
                response=response,
                context="\n".join(retrieved_texts or []) or None,
            )
            res.judge_score = s
            res.judge_reason = reason
            if s is None:
                res.failure_tags.append("judge_error")

    # combine: heuristic score is base; judge blends in when present
    base = h.score
    if res.judge_score is not None:
        # judge weighted 60%, heuristics 40%
        res.score = round(0.6 * res.judge_score + 0.4 * base, 4)
    else:
        res.score = base

    # pass/fail: all heuristics pass AND (no judge or judge score >= 0.5)
    passed = h.all_passed
    if res.judge_score is not None and res.judge_score < 0.5:
        passed = False
        res.failure_tags.append("low_judge_score")
    res.passed = passed
    return res
