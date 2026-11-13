"""OpenAI-compatible mock server for demo/evidence runs (no API keys needed).

Serves:
  POST /v1/chat/completions  — unary JSON, SSE streaming, and tool_calls
                               (returns a search_docs tool call when `tools`
                               are present and no tool result exists yet)
  POST /v1/embeddings        — OpenAI-compatible embeddings (deterministic
                               hash vector; exercises the embed stage)
  POST /retrieve             — vector-store style endpoint: accepts
                               {query, query_vector?, top_k} and returns
                               keyword-scored docs from datasets/docs.jsonl
  POST /search               — dummy docs-search endpoint for agent http tools
  GET  /health               — liveness probe

Usage: python demo/mock_server.py [port]   (default 8901)

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _chat_response(req: dict) -> dict:
    """Build a chat.completion response: tool_call -> tool result -> answer."""
    messages = req.get("messages") or []
    has_tool_result = any(m.get("role") == "tool" for m in messages)
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")

    if req.get("tools") and not has_tool_result:
        # first agent step: ask the "model" to call search_docs once
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": req.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_1",
                                "type": "function",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": json.dumps({"input": str(last_user)[:120]}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54},
        }

    if "Rate the relevance" in str(last_user):
        # rerank stage: return a high numeric score the parser can use
        content = "8"
    elif "Score the response quality" in str(last_user) or "rubric" in str(last_user).lower():
        # LLM-as-judge call: return parseable JSON score (scale 1-5 -> 0.75 norm)
        content = '{"score": 4, "reason": "mock judge: adequate"}'
    else:
        content = f"mock answer ({len(str(last_user))} chars prompt)"
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": req.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32},
    }


def _embedding_response(req: dict) -> dict:
    """Deterministic fake embeddings: hash each input into a fixed vector."""
    inputs = req.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    data = []
    for i, text in enumerate(inputs or []):
        h = 0
        for ch in str(text):
            h = (h * 31 + ord(ch)) % (2**32)
        vec = [((h >> (j * 4)) & 0xF) / 15.0 - 0.5 for j in range(8)]
        data.append({"object": "embedding", "index": i, "embedding": vec})
    return {
        "object": "list",
        "data": data,
        "model": req.get("model", "mock-embed"),
        "usage": {"prompt_tokens": sum(len(str(t).split()) for t in (inputs or [])), "total_tokens": 0},
    }


_DOCS = None


def _load_docs() -> list[dict]:
    """Load datasets/docs.jsonl once for the /retrieve mock vector store."""
    global _DOCS
    if _DOCS is None:
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "datasets", "docs.jsonl")
        _DOCS = []
        try:
            with open(path) as f:
                for raw in f:
                    line = raw.strip()
                    if line:
                        _DOCS.append(json.loads(line))
        except OSError:
            pass
    return _DOCS


def _retrieve_response(req: dict) -> dict:
    """Keyword-overlap 'vector search' over docs.jsonl (mimics a store)."""
    query = str(req.get("query") or "")
    top_k = int(req.get("top_k") or 5)
    q_tokens = set(query.lower().split())
    scored = []
    for d in _load_docs():
        text = d.get("text") or d.get("content") or ""
        overlap = len(q_tokens & set(text.lower().split()))
        scored.append((overlap, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [
        {
            "id": d.get("id"),
            "text": d.get("text") or d.get("content") or "",
            "score": round(overlap / max(len(q_tokens), 1), 4),
        }
        for overlap, d in scored[:top_k]
    ]
    return {"results": results, "matches": results}  # matches: Pinecone-style alias


def _sse_stream(req: dict):
    """Yield SSE lines for a streaming chat completion."""
    chunks = ["Hello", " from", " the", " llm-perf", " mock", "."]
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": req.get("model", "mock")}
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    for text in chunks:
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
        time.sleep(0.01)
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 20, 'completion_tokens': len(chunks), 'total_tokens': 20 + len(chunks)}})}\n\n"
    yield "data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    """Request handler for the mock OpenAI-compatible API."""

    protocol_version = "HTTP/1.1"

    def _send_json(self, obj, status: int = 200):
        """Write a JSON response."""
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        """Health probe only."""
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        """Route chat completions / search / anything else."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            req = {}

        if self.path.endswith("/chat/completions"):
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # no Content-Length: client reads until EOF — must close
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    for line in _sse_stream(req):
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client went away mid-stream — normal under load
                return
            self._send_json(_chat_response(req))
        elif self.path.endswith("/embeddings"):
            self._send_json(_embedding_response(req))
        elif self.path.endswith("/retrieve"):
            self._send_json(_retrieve_response(req))
        elif self.path.endswith("/search"):
            q = req.get("query") or req.get("input") or ""
            self._send_json(
                {
                    "results": [
                        {"title": f"doc about {str(q)[:30]}", "snippet": "relevant passage", "score": 0.9}
                    ]
                }
            )
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, *a):
        """Silence request logging."""


def main() -> None:
    """Start the mock server on the given port (default 8901)."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8901
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mock OpenAI server listening on http://127.0.0.1:{port}/v1")
    srv.serve_forever()


if __name__ == "__main__":
    main()
