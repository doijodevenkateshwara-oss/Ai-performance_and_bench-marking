"""gRPC transport for app_endpoint layer — reuses providers.grpc machinery."""

from __future__ import annotations

import time

from llm_perf.providers.base import NormalizedResponse, RequestSpec
from llm_perf.providers.grpc import GrpcAdapter


class GrpcTransport:
    def __init__(self, timeout_s: float = 120.0):
        self.timeout = timeout_s
        self._adapter = GrpcAdapter()

    def call(
        self, target_host: str, spec: RequestSpec
    ) -> tuple[NormalizedResponse | None, float, str | None]:
        spec.base_url = target_host
        spec.timeout_s = self.timeout
        start = time.monotonic()
        try:
            resp = self._adapter.call(spec)
            return resp, time.monotonic() - start, None
        except Exception as e:
            return None, time.monotonic() - start, str(e)
