"""Security policy: URL allowlisting, SSRF guards, secret redaction, config audit.

The framework executes user-supplied configuration — provider endpoints,
agent tool URLs, Python import paths, remote datasets. By default it stays
permissive (the tool exists to test internal services), but `security:` in
the YAML config can restrict egress, gate Python execution, cap dataset
sizes, and enable log/trace redaction.

Policy is a process-wide singleton set from `PerfConfig.security` via
`set_policy()` — called by the CLI and the generated locustfile so every
worker process picks it up.


Author: Venkateshwara Doijode
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.request
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter


class UrlNotAllowed(PermissionError, requests.RequestException):
    """Outbound URL rejected by the security policy.

    Subclasses requests.RequestException so provider retry/error paths treat
    a blocked URL like any other failed request instead of crashing the task.
    """


# ---------------------------------------------------------------------------
# Policy singleton
# ---------------------------------------------------------------------------

_policy = None
_warned_once: set[str] = set()


def set_policy(cfg) -> None:
    """Install a SecurityConfig (or None -> defaults)."""
    global _policy
    _policy = cfg


def get_policy():
    """Current policy; a default-constructed SecurityConfig when unset."""
    if _policy is not None:
        return _policy
    from .config import SecurityConfig

    return SecurityConfig()


def reset_policy() -> None:
    """Test helper: restore defaults and clear warn-once flags."""
    global _policy
    _policy = None
    _warned_once.clear()


# ---------------------------------------------------------------------------
# URL checks (allowlist + private-network SSRF guard)
# ---------------------------------------------------------------------------


def _host_in_allowlist(host: str, allowlist: list[str]) -> bool:
    """True when host matches an allowlist entry (suffix-aware)."""
    host = host.lower().rstrip(".")
    for entry in allowlist or []:
        e = str(entry).lower().strip()
        if not e:
            continue
        e = e.lstrip("*.")
        if host == e or host.endswith("." + e):
            return True
    return False


def _ip_blocked(ip_str: str) -> bool:
    """True when an IP literal is not globally routable."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    # block anything that isn't globally routable: loopback, RFC1918,
    # link-local (incl. 169.254.169.254 cloud metadata), multicast, reserved
    return not ip.is_global


def check_url(url: str) -> None:
    """Raise UrlNotAllowed if `url` violates the egress policy."""
    pol = get_policy()
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise UrlNotAllowed(f"unparseable URL blocked: {e}") from e
    if parsed.scheme not in ("http", "https"):
        raise UrlNotAllowed(f"URL scheme {parsed.scheme!r} not allowed (http/https only): {url}")
    host = parsed.hostname
    if not host:
        raise UrlNotAllowed(f"URL has no host: {url}")
    allowlist = getattr(pol, "url_allowlist", None) or []
    if allowlist and not _host_in_allowlist(host, allowlist):
        raise UrlNotAllowed(f"host {host!r} not in security.url_allowlist")
    if getattr(pol, "allow_private_networks", True):
        return
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        return  # unresolvable — let the HTTP client produce the real error
    for info in infos:
        ip_str = info[4][0]
        if _ip_blocked(ip_str):
            raise UrlNotAllowed(
                f"host {host!r} resolves to non-public address {ip_str} "
                "(security.allow_private_networks=false)"
            )


# ---------------------------------------------------------------------------
# Redirect-safe HTTP plumbing
#
# check_url() runs once per call site, but requests/urllib follow redirects
# transparently — a public URL could 302 to an internal address and bypass
# the policy. The helpers below re-check every hop:
#   - CheckedHTTPAdapter.send() runs for each redirect inside requests
#   - _CheckedRedirectHandler.redirect_request() runs for urllib openers
# DNS rebinding between the check and the actual connect is still possible;
# full mitigation requires connecting to the resolved IP directly.
# ---------------------------------------------------------------------------


class CheckedHTTPAdapter(HTTPAdapter):
    """requests adapter that re-validates the URL on every send (incl. redirects)."""

    def send(self, request, **kwargs):
        """Adapter.send: validate the request URL then delegate to requests."""
        check_url(request.url)
        return super().send(request, **kwargs)


def checked_session() -> requests.Session:
    """A requests.Session whose every request/redirect hop passes check_url()."""
    s = requests.Session()
    s.mount("http://", CheckedHTTPAdapter())
    s.mount("https://", CheckedHTTPAdapter())
    return s


_SHARED_SESSION: requests.Session | None = None


def shared_checked_session() -> requests.Session:
    """Process-wide checked session for one-off calls (connection reuse)."""
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        _SHARED_SESSION = checked_session()
    return _SHARED_SESSION


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib redirect handler that re-checks each hop's URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Validate the redirect target, then build the follow-up request."""
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_CHECKED_OPENER = urllib.request.build_opener(_CheckedRedirectHandler)


def checked_urlopen(req, timeout: float = 30.0):
    """urllib.request.urlopen equivalent that checks redirect targets too."""
    return _CHECKED_OPENER.open(req, timeout=timeout)


def read_response_text(resp: requests.Response, max_bytes: int | None = None) -> str:
    """Read a (preferably streamed) response body with a hard byte cap.

    Raises UrlNotAllowed when the body exceeds the cap so oversized or
    hostile endpoints can't exhaust memory. Pair with stream=True at
    request time for the cap to apply during download.
    """
    limit = max_bytes if max_bytes is not None else get_policy().http_max_response_bytes
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(65536):
        total += len(chunk)
        if total > limit:
            resp.close()
            raise UrlNotAllowed(f"response body exceeds security.http_max_response_bytes={limit}")
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_REDACTIONS = [
    # Bearer tokens first — otherwise the generic key=value rule below
    # consumes "Bearer" as the value and leaves the token exposed.
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer ***"),
    # URL userinfo: scheme://user:pass@host
    (re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1***:***@"),
    # Authorization-style headers/params
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|"
            r"password|passwd|pwd|authorization|auth|credential|private[_-]?key|client[_-]?secret)"
            r"(\s*[=:]\s*|\s+)(['\"]?)([^\s'\",}&]+)\3"
        ),
        r"\1\2\3***\3",
    ),
    # Well-known token shapes
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "ghp_***"),
    (re.compile(r"\bgho_[A-Za-z0-9]{20,}"), "gho_***"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "github_pat_***"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "xox-***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA***"),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "***PRIVATE KEY***",
    ),
]


def redact_secrets(text) -> str:
    """Scrub credential-shaped substrings from a string."""
    if not text:
        return text
    s = str(text)
    pol = get_policy()
    if not getattr(pol, "redact", True):
        return s
    for pattern, repl in _REDACTIONS:
        s = pattern.sub(repl, s)
    return s


# ---------------------------------------------------------------------------
# Python execution gate
# ---------------------------------------------------------------------------


def python_execution_allowed(kind: str) -> bool:
    """Gate for user-supplied Python (app_object python_path, function tools).

    Returns True when execution may proceed. 'warn' mode allows but emits a
    one-time loud warning; 'off' returns False.
    """
    pol = get_policy()
    mode = getattr(pol, "python_execution", "warn") or "warn"
    if mode == "off":
        return False
    if mode == "warn":
        _warn_once(
            "python_exec",
            "[llm-perf] SECURITY WARNING: config executes user-supplied Python "
            f"({kind}). Python code runs with your full privileges — only use "
            "configs you trust. Set security.python_execution: allow to "
            "acknowledge, or off to disable.",
        )
    return True


def _warn_once(key: str, msg: str) -> None:
    """Print a warning only the first time a key is seen."""
    if key in _warned_once:
        return
    _warned_once.add(key)
    print(msg)


# ---------------------------------------------------------------------------
# Untrusted-config audit
# ---------------------------------------------------------------------------


def _iter_targets(cfg):
    """Return the config's target list (empty when absent)."""
    return list(getattr(cfg, "targets", None) or [])


def audit_config(cfg) -> list[str]:
    """Return human-readable notes about powerful/risky config features."""
    notes: list[str] = []
    for t in _iter_targets(cfg):
        layer = getattr(t, "layer", "")
        name = getattr(t, "name", "?")
        if layer == "app_object" or getattr(t, "python_path", None):
            notes.append(f"target {name}: imports and executes Python ({getattr(t, 'python_path', '')})")
        if layer == "app_endpoint":
            notes.append(f"target {name}: sends requests to arbitrary endpoint {getattr(t, 'base_url', '')}")
        if getattr(t, "request_template", None):
            notes.append(f"target {name}: uses a Jinja2 request_template (arbitrary body)")
        for st in getattr(t, "pipeline", None) or []:
            vs = getattr(st, "vector_store", None) or {}
            if isinstance(vs, dict) and vs.get("type") == "http":
                notes.append(f"target {name}: pipeline retrieve calls {vs.get('url')}")
            for tool in getattr(st, "tools", None) or []:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    notes.append(f"target {name}: agent tool imports Python ({tool.get('import_path')})")
                elif isinstance(tool, dict) and tool.get("type") == "http":
                    notes.append(f"target {name}: agent tool calls {tool.get('url')}")
    for name, ds in (getattr(cfg, "datasets", None) or {}).items():
        src = getattr(ds, "source", "") or ""
        if src.startswith(("http://", "https://")):
            notes.append(f"dataset {name}: fetched from remote URL {src}")
    return notes


def warn_untrusted(cfg) -> None:
    """Print a prominent warning once per process for risky configs."""
    pol = get_policy()
    if getattr(pol, "trusted", False):
        return
    notes = audit_config(cfg)
    if not notes:
        return
    _warn_once(
        "untrusted",
        "[llm-perf] SECURITY WARNING: this configuration uses powerful features "
        "that can execute code and reach arbitrary network services:\n"
        + "\n".join(f"  - {n}" for n in notes[:10])
        + ("\n  ..." if len(notes) > 10 else "")
        + "\nOnly run configurations you trust. Set `security.trusted: true` "
        "to acknowledge and silence this warning.",
    )
