from llm_perf.providers.http_common import anthropic_sse_extractor, openai_sse_extractor, stream_sse


def _lines(payloads):
    return iter(payloads)


def test_openai_sse_extractor_delta():
    text, usage, done = openai_sse_extractor('{"choices":[{"delta":{"content":"abc"}}]}')
    assert text == "abc" and not done and usage is None


def test_openai_sse_extractor_finish():
    _text, usage, done = openai_sse_extractor(
        '{"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'
    )
    assert done is True
    assert usage["total_tokens"] == 3


def test_stream_sse_full():
    payloads = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        b"",
        b'data: {"choices":[{"delta":{"content":" world"}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        b"data: [DONE]",
    ]
    text, ttft, _usage, _done = stream_sse(_lines(payloads), openai_sse_extractor)
    assert text == "Hello world"
    assert ttft >= 0


def test_stream_sse_skips_comments_and_bad_json():
    payloads = [
        b": keep-alive",
        b"data: {not json",
        b'data: {"choices":[{"delta":{"content":"ok"}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    ]
    text, _, _, _ = stream_sse(_lines(payloads), openai_sse_extractor)
    assert text == "ok"


def test_anthropic_extractor_events():
    t, u, d = anthropic_sse_extractor('{"type":"content_block_delta","delta":{"text":"hey"}}')
    assert t == "hey" and not d
    t, u, d = anthropic_sse_extractor(
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},' '"usage":{"output_tokens":9}}'
    )
    assert u == {"output_tokens": 9} and d is True
