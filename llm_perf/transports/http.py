"""HTTP transport for app_endpoint layer and generic calls."""
from __future__ import annotations

import time
from typing import Optional

import requests


class HttpTransport:
    def __init__(self, timeout_s: float = 120.0):
        self.timeout = timeout_s
        self.session = requests.Session()

    def request(self, method: str, url: str, headers: Optional[dict] = None,
                json_body: Optional[dict] = None, data: Optional[bytes] = None,
                stream: bool = False) -> requests.Response:
        return self.session.request(method, url, headers=headers or {},
                                    json=json_body, data=data, stream=stream,
                                    timeout=self.timeout)

    def call_timed(self, method: str, url: str, **kwargs):
        """Returns (response_or_none, elapsed_s, error_str)."""
        start = time.monotonic()
        try:
            resp = self.request(method, url, **kwargs)
            elapsed = time.monotonic() - start
            return resp, elapsed, None
        except Exception as e:
            return None, time.monotonic() - start, str(e)
