"""Azure OpenAI provider — OpenAI wire format with api-key header + api-version."""
from __future__ import annotations

from .base import RequestSpec
from .openai import OpenAIAdapter


class AzureAdapter(OpenAIAdapter):
    name = "azure"

    def default_base_url(self):
        return None  # must be configured

    def default_endpoint(self, stream: bool) -> str:
        return "/chat/completions"

    def build_request(self, spec: RequestSpec):
        url, headers, body = super().build_request(spec)
        # Azure auth: api-key header, not Bearer
        headers.pop("Authorization", None)
        if spec.api_key:
            headers["api-key"] = spec.api_key
        api_version = (spec.extra.get("azure") or {}).get("api_version", "2024-10-21")
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api-version={api_version}"
        # Azure deployment name routing: use "model" param as deployment in URL path
        # variant: /openai/deployments/{deployment}/chat/completions?api-version=...
        deployment = spec.params.get("model")
        if deployment and "/deployments/" not in url:
            base = (spec.base_url or "").rstrip("/")
            url = f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            body.pop("model", None)
        return url, headers, body
