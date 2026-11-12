"""AWS Bedrock provider via boto3 Converse API (sync client in threadpool)."""

from __future__ import annotations

import concurrent.futures
import time

from .base import NormalizedResponse, RequestSpec

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
_clients: dict = {}


def _get_client(spec: RequestSpec):
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
    name = "bedrock"

    def default_base_url(self):
        return None

    def default_endpoint(self, stream: bool) -> str:
        return "converse"

    def _converse_input(self, spec: RequestSpec) -> dict:
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
        return ("bedrock://converse", {}, self._converse_input(spec))

    def parse_response(self, status, headers, body, stream_iter=None) -> NormalizedResponse:
        # body here is the converse input dict; actual call happens via call()
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
        try:
            resp = _executor.submit(client.converse, **inp).result(timeout=spec.timeout_s)
        except Exception as e:
            return NormalizedResponse(text="", raw={"error": str(e)})
        elapsed = time.monotonic() - start
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
            stage_metrics={"e2e_s": elapsed},
        )

    def _call_stream(self, client, inp, spec, start) -> NormalizedResponse:
        text_parts = []
        ttft = None
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
                        if ttft is None:
                            ttft = time.monotonic() - start
                        text_parts.append(delta)
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {}) or {}
        except Exception as e:
            return NormalizedResponse(text="", raw={"error": str(e)})
        elapsed = time.monotonic() - start
        p = int(usage.get("inputTokens") or 0)
        c = int(usage.get("outputTokens") or 0)
        return NormalizedResponse(
            text="".join(text_parts),
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=p + c,
            ttft_s=ttft or elapsed,
            model=model_id,
            raw=usage,
            stage_metrics={"e2e_s": elapsed},
        )
