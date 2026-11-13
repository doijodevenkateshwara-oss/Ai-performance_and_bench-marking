"""Tests for provider request building and response parsing.

Author: Venkateshwara Doijode
"""

import json

from llm_perf.providers import RequestSpec
from llm_perf.providers.anthropic import AnthropicAdapter
from llm_perf.providers.azure import AzureAdapter
from llm_perf.providers.base import NormalizedResponse, usage_tokens
from llm_perf.providers.gemini import GeminiAdapter
from llm_perf.providers.openai import OpenAIAdapter


def _spec(provider="openai", **kw):
    """Build a minimal RequestSpec for provider tests."""
    return RequestSpec(provider=provider, prompt="hello", params={"model": "gpt-4o-mini"}, **kw)


def test_openai_build_request():
    """OpenAI request has model/messages and auth header."""
    a = OpenAIAdapter()
    url, headers, body = a.build_request(_spec(stream=False))
    assert url.endswith("/chat/completions")
    assert headers["Content-Type"] == "application/json"
    assert body["messages"][0]["content"] == "hello"
    assert "stream" not in body


def test_openai_stream_adds_include_usage():
    """Streaming requests ask for usage in the final chunk."""
    a = OpenAIAdapter()
    _, _, body = a.build_request(_spec(stream=True))
    assert body["stream"] is True
    assert body["stream_options"]["include_usage"] is True


def test_openai_parse_response():
    """Unary chat.completion parses text/tokens/finish_reason."""
    a = OpenAIAdapter()
    body = json.dumps(
        {
            "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "model": "gpt-4o-mini",
        }
    ).encode()
    r = a.parse_response(200, {}, body)
    assert r.text == "hi there"
    assert r.prompt_tokens == 3 and r.completion_tokens == 2
    assert r.finish_reason == "stop"


def test_openai_parse_stream():
    """SSE chunks parse into text + TTFT + usage."""
    a = OpenAIAdapter()
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}',
        "data: [DONE]",
    ]
    r = a.parse_response(200, {}, b"", stream_iter=iter(chunks))
    assert r.text == "Hello world"
    assert r.total_tokens == 3
    assert r.ttft_s is not None and r.ttft_s >= 0


def test_anthropic_build_request():
    """Anthropic request uses x-api-key and version headers."""
    a = AnthropicAdapter()
    spec = RequestSpec(
        provider="anthropic",
        prompt="hi",
        params={"model": "claude-sonnet-4-5", "max_tokens": 100},
        api_key="sk-test",
    )
    url, headers, body = a.build_request(spec)
    assert url.endswith("/v1/messages")
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    assert body["max_tokens"] == 100


def test_anthropic_parse_response():
    """Unary messages response extracts text blocks + usage."""
    a = AnthropicAdapter()
    body = json.dumps(
        {
            "content": [{"type": "text", "text": "anthropic says hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 6},
            "model": "claude-sonnet-4-5",
        }
    ).encode()
    r = a.parse_response(200, {}, body)
    assert r.text == "anthropic says hi"
    assert r.prompt_tokens == 4 and r.completion_tokens == 6


def test_anthropic_parse_stream():
    """Anthropic SSE events parse into text + usage."""
    a = AnthropicAdapter()
    lines = [
        "event: message_start",
        'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
        'data: {"type":"message_stop"}',
    ]
    r = a.parse_response(200, {}, b"", stream_iter=iter(lines))
    assert r.text == "Hi!"
    assert r.prompt_tokens == 5
    assert r.completion_tokens == 2


def test_azure_url_and_headers():
    """Azure URL gets api-version and api-key header."""
    a = AzureAdapter()
    spec = RequestSpec(
        provider="azure",
        prompt="hi",
        params={"model": "my-deployment", "max_tokens": 10},
        api_key="azkey",
        base_url="https://res.openai.azure.com",
        extra={"azure": {"api_version": "2024-10-21"}},
    )
    url, headers, body = a.build_request(spec)
    assert "api-version=2024-10-21" in url
    assert headers["api-key"] == "azkey"
    assert "Authorization" not in headers
    assert "/deployments/my-deployment/" in url
    assert "model" not in body


def test_gemini_build_request():
    """Gemini request uses x-goog-api-key, model path, generationConfig."""
    a = GeminiAdapter()
    spec = RequestSpec(
        provider="gemini",
        prompt="hi",
        params={
            "model": "gemini-2.0-flash",
            "max_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
            "system": "be terse",
        },
        api_key="gkey",
        stream=False,
    )
    url, headers, body = a.build_request(spec)
    assert url.endswith("/v1beta/models/gemini-2.0-flash:generateContent")
    assert headers["x-goog-api-key"] == "gkey"
    assert "Authorization" not in headers
    assert body["contents"][0]["parts"][0]["text"] == "hi"
    assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
    gen = body["generationConfig"]
    assert gen["maxOutputTokens"] == 64 and gen["temperature"] == 0.2 and gen["topP"] == 0.9


def test_gemini_stream_url_and_endpoint_override():
    """Streaming hits streamGenerateContent?alt=sse; custom endpoint supports {model}."""
    a = GeminiAdapter()
    spec = _spec(provider="gemini", stream=True)
    url, _, _ = a.build_request(spec)
    assert ":streamGenerateContent?alt=sse" in url
    # Vertex-style endpoint override with {model} placeholder
    spec2 = RequestSpec(
        provider="gemini",
        prompt="hi",
        params={"model": "gemini-2.0-flash"},
        base_url="https://us-central1-aiplatform.googleapis.com",
        endpoint="/v1/projects/p/locations/us-central1/publishers/google/models/{model}:generateContent",
    )
    url2, _, _ = a.build_request(spec2)
    assert "publishers/google/models/gemini-2.0-flash:generateContent" in url2


def test_gemini_parse_response():
    """Unary generateContent parses candidates text + usageMetadata."""
    a = GeminiAdapter()
    body = json.dumps(
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "gemini says hi"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 5,
                "totalTokenCount": 12,
            },
        }
    ).encode()
    r = a.parse_response(200, {}, body)
    assert r.text == "gemini says hi"
    assert r.prompt_tokens == 7 and r.completion_tokens == 5 and r.total_tokens == 12
    assert r.finish_reason == "STOP"


def test_gemini_parse_stream():
    """SSE chunks parse text deltas, skip thought parts, merge usage."""
    a = GeminiAdapter()
    lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":" thinking","thought":true},{"text":" world"}]}}]}',
        'data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}',
    ]
    r = a.parse_response(200, {}, b"", stream_iter=iter(lines))
    assert r.text == "Hello world"
    assert r.prompt_tokens == 3 and r.completion_tokens == 2
    assert r.ttft_s is not None


def test_usage_tokens_variants():
    """usage_tokens handles OpenAI and Bedrock-style keys."""
    assert usage_tokens({"prompt_tokens": 1, "completion_tokens": 2}) == (1, 2, 3)
    assert usage_tokens({"input_tokens": 5, "output_tokens": 7}) == (5, 7, 12)
    assert usage_tokens(None) == (0, 0, 0)


def test_normalized_response_tokens():
    """NormalizedResponse.tokens sums prompt+completion."""
    r = NormalizedResponse(prompt_tokens=3, completion_tokens=4)
    assert r.tokens == 7
