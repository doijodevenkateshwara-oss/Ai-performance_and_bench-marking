"""OpenTelemetry instruments for llm_perf (GenAI semconv + custom metrics)."""

from __future__ import annotations

from functools import lru_cache

from .setup import get_meter


@lru_cache(maxsize=1)
def _instruments():
    m = get_meter("llm_perf")
    return {
        "request_duration": m.create_histogram(
            "llm_perf.request.duration", unit="s", description="End-to-end request duration"
        ),
        "request_total": m.create_counter("llm_perf.request.total", description="Total requests"),
        "ttft": m.create_histogram("llm_perf.ttft", unit="s", description="Time to first token"),
        "tokens": m.create_counter("llm_perf.tokens", description="Token usage"),
        "tokens_per_second": m.create_histogram(
            "llm_perf.tokens_per_second",
            unit="{token}/s",
            description="Per-request output tokens per second distribution",
        ),
        "input_tokens_per_second": m.create_histogram(
            "llm_perf.input_tokens_per_second",
            unit="{token}/s",
            description="Per-request input (prompt) tokens per second distribution",
        ),
        "tpot": m.create_histogram(
            "llm_perf.tpot", unit="s", description="Time per output token (decode cadence)"
        ),
        "generation_duration": m.create_histogram(
            "llm_perf.generation.duration",
            unit="s",
            description="First token to last token (decode phase only)",
        ),
        "inter_token_latency": m.create_histogram(
            "llm_perf.inter_token_latency",
            unit="s",
            description="Mean inter-chunk latency during streaming",
        ),
        "queue_wait": m.create_histogram(
            "llm_perf.queue_wait",
            unit="s",
            description="Task start to request dispatch (pre-request overhead)",
        ),
        "retries": m.create_counter("llm_perf.retries", description="Retry attempts on transient failures"),
        "timeouts": m.create_counter("llm_perf.timeouts", description="Requests that timed out"),
        "users": m.create_up_down_counter("llm_perf.users", description="Active virtual users"),
        "stage_duration": m.create_histogram(
            "llm_perf.stage.duration",
            unit="s",
            description="Per-stage duration (retrieve/rerank/generate/agent_step)",
        ),
        "quality_score": m.create_histogram(
            "llm_perf.quality.score", unit="1", description="Quality score 0..1"
        ),
        "quality_pass": m.create_counter(
            "llm_perf.quality.pass", description="Quality check pass/fail counts"
        ),
        "context_recall": m.create_histogram(
            "llm_perf.rag.context_recall", unit="1", description="RAG context recall vs expected_context"
        ),
        "cost": m.create_counter("llm_perf.cost.USD", unit="USD", description="Estimated cost"),
        "threshold_pass": m.create_gauge(
            "llm_perf.threshold.pass", unit="1", description="Threshold rule pass state"
        ),
        "agent_steps": m.create_histogram(
            "llm_perf.agent.steps", unit="{step}", description="Agent LLM steps per run"
        ),
        "agent_tool_calls": m.create_counter(
            "llm_perf.agent.tool_calls", description="Agent tool invocations"
        ),
        "agent_tool_errors": m.create_counter(
            "llm_perf.agent.tool_errors", description="Agent tool calls that errored"
        ),
        "agent_loops": m.create_counter(
            "llm_perf.agent.loops", description="Agent runs flagged for repeated tool-call loops"
        ),
        "agent_task_success": m.create_counter(
            "llm_perf.agent.task_success", description="Agent task success/failure counts"
        ),
    }


def instruments():
    return _instruments()


def attrs(**kw) -> dict:
    return {k: v for k, v in kw.items() if v is not None}


GENAI_ATTRS = {
    "gen_ai.system": "gen_ai.system",
    "gen_ai.request.model": "gen_ai.request.model",
    "gen_ai.usage.input_tokens": "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens": "gen_ai.usage.output_tokens",
    "gen_ai.response.finish_reasons": "gen_ai.response.finish_reasons",
    "gen_ai.operation.name": "gen_ai.operation.name",
}
