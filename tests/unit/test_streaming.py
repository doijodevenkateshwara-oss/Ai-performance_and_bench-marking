"""Tests for SSE parsing and streaming timing statistics.

Author: Venkateshwara Doijode
"""

from llm_perf.providers.http_common import (
    anthropic_sse_extractor,
    openai_sse_extractor,
    stream_sse,
    stream_timing_stats,
)


def _lines(payloads):
    """Encode strings as fake SSE byte lines."""
    return iter(payloads)


def test_openai_sse_extractor_delta():
    """Content deltas extract text from chunks."""
    text, usage, done = openai_sse_extractor('{"choices":[{"delta":{"content":"abc"}}]}')
    assert text == "abc" and not done and usage is None


def test_openai_sse_extractor_finish():
    """finish_reason marks the stream done."""
    _text, usage, done = openai_sse_extractor(
        '{"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'
    )
    assert done is True
    assert usage["total_tokens"] == 3


def test_stream_sse_full():
    """Full stream accumulates text, TTFT, usage, offsets."""
    payloads = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        b"",
        b'data: {"choices":[{"delta":{"content":" world"}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        b"data: [DONE]",
    ]
    sr = stream_sse(_lines(payloads), openai_sse_extractor)
    assert sr.text == "Hello world"
    assert sr.ttft_s >= 0
    assert sr.completed is True
    assert len(sr.chunk_offsets_s) == 2
    assert sr.e2e_s >= sr.ttft_s


def test_stream_sse_skips_comments_and_bad_json():
    """Comments and malformed payloads are skipped."""
    payloads = [
        b": keep-alive",
        b"data: {not json",
        b'data: {"choices":[{"delta":{"content":"ok"}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    ]
    sr = stream_sse(_lines(payloads), openai_sse_extractor)
    assert sr.text == "ok"


def test_stream_timing_stats():
    # two chunks 100ms apart, 5 completion tokens -> generation 0.1s, tpot ~0.025s
    """TPOT/ITL stats derive from chunk offsets."""
    offs = [0.5, 0.6]
    stats = stream_timing_stats(offs, completion_tokens=5)
    assert abs(stats["generation_s"] - 0.1) < 1e-9
    assert abs(stats["tpot_s"] - 0.1 / 4) < 1e-9
    assert abs(stats["itl_avg_ms"] - 100.0) < 1e-6
    assert abs(stats["itl_p50_ms"] - 100.0) < 1e-6
    assert abs(stats["itl_p95_ms"] - 100.0) < 1e-6
    assert abs(stats["itl_max_ms"] - 100.0) < 1e-6
    assert stats["chunks"] == 2


def test_stream_timing_stats_single_chunk():
    """Single-chunk streams yield null timing stats."""
    stats = stream_timing_stats([0.3], completion_tokens=10)
    assert stats["generation_s"] is None
    assert stats["tpot_s"] is None
    assert stats["itl_avg_ms"] is None
    assert stats["chunks"] == 1


def test_stream_sse_first_token_only():
    """first_token_only stops after the first content chunk."""
    payloads = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}',
        b'data: {"choices":[{"delta":{"content":"b"}}]}',
        b"data: [DONE]",
    ]
    sr = stream_sse(_lines(payloads), openai_sse_extractor, first_token_only=True)
    assert sr.text == "a"
    assert sr.completed is False
    assert len(sr.chunk_offsets_s) == 1


def test_anthropic_extractor_events():
    """Anthropic event types map to deltas/usage/done."""
    t, u, d = anthropic_sse_extractor('{"type":"content_block_delta","delta":{"text":"hey"}}')
    assert t == "hey" and not d
    t, u, d = anthropic_sse_extractor(
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},' '"usage":{"output_tokens":9}}'
    )
    assert u == {"output_tokens": 9} and d is True
