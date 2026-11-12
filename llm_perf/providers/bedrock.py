"""AWS Bedrock provider via boto3 Converse API (sync client in threadpool).

Author: Venkateshwara Doijode
"""

from __future__ import annotations

import concurrent.futures
import time

from .base import NormalizedResponse, RequestSpec
from .http_common import stream_timing_stats

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
_clients: dict = {}


def _get_client(spec: RequestSpec):
    """Return a cached boto3 bedrock-runtime client per aws config."""
    key = (
        spec.extra.get("aws", {}).get("region"),
        spec.extra.get("aws", {}).get("auth"),
        spec.extra.get("aws", {}).get("profile"),
    )
    if key not in _clients:
        import boto3

        aws_cfg = spec.extra.get("aws", {})
        kwargs = {}
        if aws_cfg.get("region"):
            kwargs["region_name"] = aws_cfg["region"]
        if aws_cfg.get("profile"):
            kwargs["profile_name"] = aws_cfg["profile"]
        _clients[key] = boto3.client("bedrock-runtime", **kwargs)
    return _clients[key]


class BedrockAdapter:
    """AWS Bedrock Converse API adapter (boto3 in a threadpool)."""

    name = "bedrock"

    def default_base_url(self):
        """None — Bedrock is not HTTP-configured."""
        return None

    def default_endpoint(self, stream: bool) -> str:
        """Logical endpoint label ('converse')."""
        return "converse"

    def _converse_input(self, spec: RequestSpec) -> dict:
        """Build the Converse API request dict from a RequestSpec."""
        params = spec.params
        model_id = params.get("model") or params.get("model_id")
        if not model_id:
            raise ValueError("bedrock target requires params.model (model/inference-profile id)")
        body = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": spec.prompt}]}],
            "inferenceConfig": {},
        }
        for k, bk in (
            ("max_tokens", "maxTokens"),
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("stop", "stopSequences"),
        ):
            if k in params:
                body["inferenceConfig"][bk] = params[k]
        if "system" in params:
            body["system"] = [{"text": params["system"]}]
        return body

    def build_request(self, spec: RequestSpec):
        # boto3-based; no HTTP request to build
        """Return a synthetic request descriptor (real call via call())."""
        return ("bedrock://converse", {}, self._converse_input(spec))

    def parse_response(self, status, headers, body, stream_iter=None) -> NormalizedResponse:
        # body here is the converse input dict; actual call happens via call()
        """Unused — Bedrock executes via call(); raises NotImplementedError."""
        raise NotImplementedError("use BedrockAdapter.call()")

    def call(self, spec: RequestSpec) -> NormalizedResponse:
        """Execute the converse call in a worker thread (Locust-safe)."""
        inp = self._converse_input(spec)
        client = _get_client(spec)
        start = time.monotonic()
        if spec.stream:
            return self._call_stream(client, inp, spec, start)
        return self._call_unary(client, inp, spec, start)

    def _call_unary(self, client, inp, spec, start) -> NormalizedResponse:
        """Run a blocking converse call in the threadpool and normalize."""
        try:
            resp = _executor.submit(client.converse, **inp).result(timeout=spec.timeout_s)
        except Exception as e:
            return NormalizedResponse(text="", raw={"error": str(e)})
        out = resp.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in out if isinstance(b, dict))
        usage = resp.get("usage") or {}
        p = int(usage.get("inputTokens") or 0)
        c = int(usage.get("outputTokens") or 0)
        return NormalizedResponse(
            text=text,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=p + c,
            finish_reason=(resp.get("stopReason")),
            model=inp["modelId"],
            raw=resp,
        )

    def _call_stream(self, client, inp, spec, start) -> NormalizedResponse:
        """Consume converse_stream events; collect text/usage/timing."""
        text_parts = []
        ttft = None
        chunk_offsets: list[float] = []
        usage = {}
        model_id = inp.pop("modelId")
        inference = inp.pop("inferenceConfig", {})
        try:
            stream = client.converse_stream(
                modelId=model_id,
                messages=inp["messages"],
                inferenceConfig=inference or None,
                **({"system": inp["system"]} if "system" in inp else {}),
            )
            for event in stream.get("stream"):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {}).get("text", "")
                    if delta:
                        now = time.monotonic() - start
                        if ttft is None:
                            ttft = now
                        text_parts.append(delta)
                        chunk_offsets.append(now)
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {}) or {}
        except Exception as e:
            return NormalizedResponse(text="", raw={"error": str(e)})
        elapsed = time.monotonic() - start
        p = int(usage.get("inputTokens") or 0)
        c = int(usage.get("outputTokens") or 0)
        timing = stream_timing_stats(chunk_offsets, c)
        return NormalizedResponse(
            text="".join(text_parts),
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=p + c,
            ttft_s=ttft or elapsed,
            tpot_s=timing["tpot_s"],
            generation_s=timing["generation_s"],
            stream_stats=timing,
            model=model_id,
            raw=usage,
        )
