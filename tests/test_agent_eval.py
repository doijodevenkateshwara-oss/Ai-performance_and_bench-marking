"""Agent-stage evaluation: success, tool correctness, loop/failure detection."""

from __future__ import annotations

import pytest

from llm_perf.config import AgentSuccessConfig, PipelineStage, PricingConfig
from llm_perf.pipelines import engine
from llm_perf.pipelines.engine import StageContext, run_pipeline, run_stage_agent
from llm_perf.providers.base import NormalizedResponse


def _tc(name: str, arguments: str = '{"input": "q"}', tc_id: str = "call_1") -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _resp(text: str = "", tool_calls=None, ptok: int = 10, ctok: int = 5) -> NormalizedResponse:
    return NormalizedResponse(
        text=text,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        total_tokens=ptok + ctok,
        tool_calls=tool_calls,
    )


def _stage(**kw) -> PipelineStage:
    return PipelineStage(stage="agent", llm={"provider": "openai", "model": "gpt-4o"}, **kw)


def _ctx(row=None, pricing=None) -> StageContext:
    return StageContext("question?", row or {}, "sc", {}, pricing=pricing)


@pytest.fixture
def patch_llm(monkeypatch):
    calls = []

    def _install(script):
        it = iter(script)

        def fake(spec):
            calls.append(spec)
            try:
                return next(it)
            except StopIteration:
                return _resp("done")

        monkeypatch.setattr(engine, "call_llm", fake)
        return calls

    return _install


def test_agent_happy_path_success(patch_llm):
    patch_llm([_resp(tool_calls=[_tc("search", tc_id="c1")]), _resp("final answer here")])
    stage = _stage(tools=[{"name": "search", "type": "function", "import_path": "tests.agent_tools:echo"}])
    ctx = _ctx()
    run_stage_agent(stage, ctx)
    ev = ctx.agent_eval
    assert ev["terminated"] == "final_answer"
    assert ev["tool_calls"] == 1
    assert ev["tool_errors"] == 0
    assert ev["success"] is True
    assert ctx.stage_metrics["agent_steps"] == 2


def test_agent_loop_detection(patch_llm):
    # model repeats the identical tool call forever
    patch_llm([_resp(tool_calls=[_tc("search", tc_id="c1")])] * 10)
    stage = _stage(
        max_iterations=6,
        loop_repeat_threshold=2,
        tools=[{"name": "search", "type": "function", "import_path": "tests.agent_tools:echo"}],
    )
    ctx = _ctx()
    run_stage_agent(stage, ctx)
    ev = ctx.agent_eval
    assert ev["terminated"] == "max_iterations"
    assert ev["loops"] >= 1
    assert ev["success"] is False  # default on_max_iterations=fail


def test_agent_max_iterations_pass_config(patch_llm):
    patch_llm([_resp(tool_calls=[_tc("search", tc_id="c1")])] * 10)
    stage = _stage(
        max_iterations=3,
        tools=[{"name": "search", "type": "function", "import_path": "tests.agent_tools:echo"}],
        success=AgentSuccessConfig(on_max_iterations="pass"),
    )
    ctx = _ctx()
    run_stage_agent(stage, ctx)
    assert ctx.agent_eval["success"] is True


def test_agent_unknown_tool(patch_llm):
    patch_llm([_resp(tool_calls=[_tc("nonexistent", tc_id="c1")]), _resp("answer")])
    stage = _stage(tools=[])
    ctx = _ctx()
    run_stage_agent(stage, ctx)
    ev = ctx.agent_eval
    assert ev["unknown_tools"] == 1
    assert ev["tool_errors"] == 1  # executor None -> "error: unknown tool"


def test_agent_expected_field_match(patch_llm):
    patch_llm([_resp("The capital of France is Paris.")])
    stage = _stage(success=AgentSuccessConfig(expected_field="expected", match="contains"))
    ctx = _ctx(row={"expected": "Paris"})
    run_stage_agent(stage, ctx)
    assert ctx.agent_eval["success"] is True

    stage2 = _stage(success=AgentSuccessConfig(expected_field="expected", match="exact"))
    ctx2 = _ctx(row={"expected": "Paris"})
    run_stage_agent(stage2, ctx2)
    assert ctx2.agent_eval["success"] is False  # not an exact match


def test_agent_require_tool_use(patch_llm):
    patch_llm([_resp("answer without tools")])
    stage = _stage(success=AgentSuccessConfig(require_tool_use=True))
    ctx = _ctx()
    run_stage_agent(stage, ctx)
    assert ctx.agent_eval["success"] is False


def test_agent_empty_final_fails(patch_llm):
    patch_llm([_resp("")])
    ctx = _ctx()
    run_stage_agent(_stage(), ctx)
    assert ctx.agent_eval["final_empty"] is True
    assert ctx.agent_eval["success"] is False


def test_agent_cost_cap(patch_llm):
    # expensive pricing -> first step already exceeds cap
    patch_llm([_resp(tool_calls=[_tc("search", tc_id="c1")])] * 5)
    stage = _stage(
        max_iterations=6,
        max_cost_usd=0.0001,
        tools=[{"name": "search", "type": "function", "import_path": "tests.agent_tools:echo"}],
    )
    pricing = PricingConfig(input_per_1k=1.0, output_per_1k=1.0)  # 10+5 tokens -> $0.015/step
    ctx = _ctx(pricing=pricing)
    run_stage_agent(stage, ctx)
    assert ctx.agent_eval["terminated"] == "cost_cap"


def test_run_pipeline_propagates_agent_metrics(patch_llm):
    patch_llm([_resp("done answer")])
    stage = _stage()
    ctx = run_pipeline([stage], "q?", {}, "sc", {}, pricing=None)
    assert ctx.agent_eval["terminated"] == "final_answer"
    assert ctx.stage_metrics["agent_steps"] == 1
    assert ctx.stage_metrics["agent_success"] == 1.0
