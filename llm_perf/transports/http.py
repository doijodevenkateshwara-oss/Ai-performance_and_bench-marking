"""HTTP transport for app_endpoint layer and generic calls."""

from __future__ import annotations

import time

import requests


class HttpTransport:
    def __init__(self, timeout_s: float = 120.0):
        self.timeout = timeout_s
        self.session = requests.Session()

    def request(
        self,
        method: str,
        url: str,
        headers: dict | None = None,
        json_body: dict | None = None,
        data: bytes | None = None,
        stream: bool = False,
    ) -> requests.Response:
        from ..security import check_url

        check_url(url)
        return self.session.request(
            method, url, headers=headers or {}, json=json_body, data=data, stream=stream, timeout=self.timeout
        )

    def call_timed(self, method: str, url: str, **kwargs):
        """Returns (response_or_none, elapsed_s, error_str)."""
        start = time.monotonic()
        try:
            resp = self.request(method, url, **kwargs)
        except Exception as e:
            return None, time.monotonic() - start, str(e)
        else:
            return resp, time.monotonic() - start, None
