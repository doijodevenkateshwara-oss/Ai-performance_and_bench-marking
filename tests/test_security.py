"""Security policy tests: URL guards, redaction, python gate, dataset limits."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_perf import security
from llm_perf.config import SecurityConfig
from llm_perf.security import UrlNotAllowed, check_url, python_execution_allowed, redact_secrets


@pytest.fixture(autouse=True)
def _reset_policy():
    security.reset_policy()
    yield
    security.reset_policy()


def _policy(**kw) -> SecurityConfig:
    return SecurityConfig(**kw)


# ---------------------------------------------------------------------------
# check_url
# ---------------------------------------------------------------------------


def test_default_policy_allows_http_and_localhost():
    security.set_policy(_policy())
    check_url("http://localhost:8080/api")
    check_url("https://api.openai.com/v1/chat")


def test_non_http_schemes_blocked():
    security.set_policy(_policy())
    for url in ("file:///etc/passwd", "ftp://x/y", "gopher://x"):
        with pytest.raises(UrlNotAllowed):
            check_url(url)


def test_allowlist_exact_and_suffix():
    security.set_policy(_policy(url_allowlist=["api.example.com", "*.corp.internal"]))
    check_url("https://api.example.com/v1")
    check_url("https://x.corp.internal/hook")
    with pytest.raises(UrlNotAllowed):
        check_url("https://evil.com/x")
    with pytest.raises(UrlNotAllowed):
        check_url("https://notexample.com/x")  # suffix must be dot-boundary


def test_private_networks_blocked_when_disabled(monkeypatch):
    security.set_policy(_policy(allow_private_networks=False))

    def fake_gai(host, port, *a, **kw):
        return [(None, None, None, None, ("127.0.0.1", port))]

    monkeypatch.setattr(security.socket, "getaddrinfo", fake_gai)
    with pytest.raises(UrlNotAllowed, match="non-public"):
        check_url("http://internal.corp.local/metrics")

    def fake_gai_public(host, port, *a, **kw):
        return [(None, None, None, None, ("93.184.216.34", port))]

    monkeypatch.setattr(security.socket, "getaddrinfo", fake_gai_public)
    check_url("https://api.example.com/v1")  # public IP -> allowed


def test_private_networks_allowed_by_default():
    security.set_policy(_policy())
    check_url("http://localhost:9090/metrics")
    check_url("http://169.254.169.254/latest/meta-data")  # default: permitted


def test_unresolvable_host_allowed_when_private_blocked(monkeypatch):
    security.set_policy(_policy(allow_private_networks=False))

    def boom(host, port, *a, **kw):
        raise security.socket.gaierror("no such host")

    monkeypatch.setattr(security.socket, "getaddrinfo", boom)
    check_url("http://does-not-exist.invalid/x")  # can't prove internal -> allow


# ---------------------------------------------------------------------------
# redact_secrets
# ---------------------------------------------------------------------------


def test_redact_bearer_and_params():
    security.set_policy(_policy())
    out = redact_secrets("GET /x Authorization: Bearer abcdef1234567890 ok api_key=sk-live-999")
    assert "abcdef1234567890" not in out
    assert "sk-live-999" not in out
    assert "***" in out


def test_redact_url_userinfo_and_tokens():
    security.set_policy(_policy())
    out = redact_secrets("https://user:passw0rd@host/path ghp_" + "a" * 30 + " sk-" + "b" * 20)
    assert "passw0rd" not in out
    assert "ghp_" + "a" * 30 not in out
    assert "sk-" + "b" * 20 not in out


def test_redact_disabled_by_policy():
    security.set_policy(_policy(redact=False))
    s = "api_key=supersecretvalue"
    assert redact_secrets(s) == s


# ---------------------------------------------------------------------------
# python_execution gate
# ---------------------------------------------------------------------------


def test_python_execution_modes(capsys):
    security.set_policy(_policy(python_execution="off"))
    assert python_execution_allowed("test") is False

    security.set_policy(_policy(python_execution="warn"))
    assert python_execution_allowed("test") is True
    assert python_execution_allowed("test") is True  # warns once
    assert capsys.readouterr().out.count("SECURITY WARNING") == 1

    security.reset_policy()
    security.set_policy(_policy(python_execution="allow"))
    assert python_execution_allowed("test") is True
    assert "SECURITY WARNING" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dataset limits
# ---------------------------------------------------------------------------


def test_dataset_max_bytes(tmp_path):
    from llm_perf.datasets import _read_text

    f = tmp_path / "big.jsonl"
    f.write_text("x" * 100)
    security.set_policy(_policy(dataset_max_bytes=50))
    with pytest.raises(ValueError, match="dataset_max_bytes"):
        _read_text(str(f))


def test_dataset_max_rows(tmp_path):
    from llm_perf.datasets import _read_rows

    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join(json.dumps({"q": i}) for i in range(10)))
    security.set_policy(_policy(dataset_max_rows=5))
    with pytest.raises(ValueError, match="dataset_max_rows"):
        _read_rows(str(f), "jsonl")


def test_dataset_http_fetch_and_limits():
    """HTTP dataset source: allowlist blocks, size cap enforced."""

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"x" * 200
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/data.jsonl"
    try:
        from llm_perf.datasets import _read_text

        # private networks allowed by default -> fetch works, size cap applies
        security.set_policy(_policy(dataset_max_bytes=50))
        with pytest.raises(ValueError, match="dataset_max_bytes"):
            _read_text(url)

        # block private nets -> URL rejected before fetch
        security.set_policy(_policy(allow_private_networks=False))
        with pytest.raises(UrlNotAllowed):
            _read_text(url)
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# agent tool / pipeline enforcement
# ---------------------------------------------------------------------------


def test_agent_http_tool_blocked_by_allowlist():
    from llm_perf.pipelines.engine import _execute_tool

    security.set_policy(_policy(url_allowlist=["allowed.example.com"]))
    executors = {"t": ("http", {"url": "http://internal.local/hook", "method": "POST"})}
    text, _ = _execute_tool(executors["t"], "t", {"input": "hi"}, None)
    assert text.startswith("error:")
    assert "url_allowlist" in text or "not allowed" in text.lower() or "blocked" in text.lower()


def test_agent_function_tool_blocked_when_off():
    from llm_perf.pipelines.engine import _execute_tool

    security.set_policy(_policy(python_execution="off"))
    executors = {"f": ("function", {"import_path": "os:getcwd"})}
    text, _ = _execute_tool(executors["f"], "f", {}, None)
    assert text.startswith("error:")
    assert "python execution disabled" in text


# ---------------------------------------------------------------------------
# audit / warnings
# ---------------------------------------------------------------------------


def test_audit_config_detects_powerful_features(capsys):
    from types import SimpleNamespace

    from llm_perf.security import audit_config, warn_untrusted

    cfg = SimpleNamespace(
        targets=[
            SimpleNamespace(
                name="app",
                layer="app_object",
                python_path="mymod:build",
                pipeline=None,
                request_template=None,
                base_url=None,
            ),
            SimpleNamespace(
                name="ep",
                layer="app_endpoint",
                base_url="http://internal:8080",
                python_path=None,
                pipeline=[
                    SimpleNamespace(
                        vector_store=None,
                        tools=[{"type": "function", "import_path": "x:y"}],
                    )
                ],
                request_template=None,
            ),
        ],
        datasets={"d": SimpleNamespace(source="https://example.com/d.jsonl")},
    )
    notes = audit_config(cfg)
    assert any("Python" in n for n in notes)
    assert any("agent tool imports Python" in n for n in notes)
    assert any("remote URL" in n for n in notes)

    security.set_policy(_policy())
    warn_untrusted(cfg)
    assert "SECURITY WARNING" in capsys.readouterr().out

    capsys.readouterr()
    security.set_policy(_policy(trusted=True))
    warn_untrusted(cfg)
    assert capsys.readouterr().out == ""
