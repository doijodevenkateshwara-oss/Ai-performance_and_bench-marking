"""Pydantic configuration models for llm-perf YAML configs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

class OtelConfig(BaseModel):
    enabled: bool = True
    endpoint: Optional[str] = None  # OTLP gRPC endpoint; falls back to env/default
    service_name: str = "llm-perf"
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    genai_semconv: bool = True


class PrometheusConfig(BaseModel):
    """Deprecated alias kept for backwards compat; OTel is the metrics path."""
    enabled: bool = False
    port: int = 9090
    interval_s: int = 15


class ObservabilityConfig(BaseModel):
    otel: OtelConfig = Field(default_factory=OtelConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DatasetSample(BaseModel):
    strategy: Literal["first", "random", "weighted"] = "first"
    count: Optional[int] = None
    seed: int = 42
    weights: Optional[dict[str, float]] = None  # tag/column value -> weight


class DatasetConfig(BaseModel):
    source: str  # path, URL, s3://, gs://
    format: Optional[Literal["yaml", "csv", "jsonl"]] = None  # auto-detect
    columns: dict[str, str] = Field(default_factory=dict)  # logical -> source field
    sample: Optional[DatasetSample] = None
    vars: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _source_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("dataset source must not be empty")
        return v


# ---------------------------------------------------------------------------
# Targets / layers
# ---------------------------------------------------------------------------

class AzureConfig(BaseModel):
    api_version: str = "2024-10-21"
    deployment_field: str = "model"


class AwsConfig(BaseModel):
    region: Optional[str] = None
    auth: Literal["env", "profile", "iam_role"] = "env"
    profile: Optional[str] = None


class GrpcConfig(BaseModel):
    proto_path: Optional[str] = None  # None -> bundled triton proto
    service: str = "inference.GRPCInferenceService"
    method: str = "ModelInfer"
    tls: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)
    request_fields: dict[str, str] = Field(default_factory=dict)  # logical->proto field
    response_fields: dict[str, str] = Field(default_factory=dict)


class ProviderParams(BaseModel):
    """Provider-specific extra config, kept flexible."""
    model_config = {"extra": "allow"}


class BaseTarget(BaseModel):
    name: str
    layer: Literal["llm_api", "pipeline", "app_endpoint", "app_object"] = "llm_api"
    weight: float = 1.0  # used when referenced from scenarios; also default scenario weight
    think_time: Optional[tuple[float, float]] = None
    timeout_s: float = 120.0
    tags: dict[str, str] = Field(default_factory=dict)
    vars: dict[str, list[Any]] = Field(default_factory=dict)  # {var} template pools

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid target name: {v!r}")
        return v


class LlmApiTarget(BaseTarget):
    layer: Literal["llm_api"] = "llm_api"
    provider: Literal["openai", "anthropic", "azure", "bedrock", "http", "grpc"]
    transport: Literal["http", "grpc"] = "http"
    base_url: Optional[str] = None
    endpoint: Optional[str] = None  # http path or grpc method override
    api_key_env: Optional[str] = None
    azure: Optional[AzureConfig] = None
    aws: Optional[AwsConfig] = None
    grpc: Optional[GrpcConfig] = None
    params: dict[str, Any] = Field(default_factory=dict)  # model, max_tokens, ...
    stream: bool = False
    stream_mode: Literal["full", "first_token"] = "full"
    request_template: Optional[str] = None  # jinja2 for provider=http
    response_mapping: dict[str, str] = Field(default_factory=dict)
    dataset: Optional[str] = None
    prompts: Optional[list[str]] = None

    @model_validator(mode="after")
    def _check_endpoint(self):
        if self.provider in ("openai", "anthropic", "azure", "bedrock") and not self.base_url:
            # sensible defaults
            defaults = {
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com",
                "azure": None,  # must be provided
                "bedrock": None,
            }
            self.base_url = defaults.get(self.provider)
        if self.transport == "http" and self.provider != "grpc" and not self.base_url and self.provider in ("azure",):
            raise ValueError("azure target requires base_url (https://<resource>.openai.azure.com)")
        return self


class PipelineStage(BaseModel):
    stage: Literal["retrieve", "rerank", "generate", "agent"]
    # retrieve
    vector_store: Optional[dict[str, Any]] = None
    top_k: int = 5
    # rerank
    method: Optional[Literal["llm", "cross_encoder"]] = None
    # generate / agent llm ref
    llm: Optional[dict[str, Any]] = None
    template: Optional[str] = None
    # agent
    max_iterations: int = 6
    max_cost_usd: Optional[float] = None
    tools: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class PipelineTarget(BaseTarget):
    layer: Literal["pipeline"] = "pipeline"
    dataset: str
    pipeline: list[PipelineStage]


class AppEndpointTarget(BaseTarget):
    layer: Literal["app_endpoint"] = "app_endpoint"
    transport: Literal["http", "grpc"] = "http"
    base_url: str
    endpoint: Optional[str] = None
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    request_template: Optional[str] = None
    response_mapping: dict[str, str] = Field(default_factory=dict)
    grpc: Optional[GrpcConfig] = None
    dataset: Optional[str] = None
    prompts: Optional[list[str]] = None
    api_key_env: Optional[str] = None


class AppObjectTarget(BaseTarget):
    layer: Literal["app_object"] = "app_object"
    framework: Literal["langchain", "llamaindex", "generic"] = "generic"
    python_path: str  # "module.path:factory_fn"
    call: str = "ainvoke"  # ainvoke|invoke|query|aquery|custom
    init_args: dict[str, Any] = Field(default_factory=dict)
    dataset: Optional[str] = None
    prompts: Optional[list[str]] = None


Target = Union[LlmApiTarget, PipelineTarget, AppEndpointTarget, AppObjectTarget]


# ---------------------------------------------------------------------------
# Scenarios / load / quality / thresholds / report
# ---------------------------------------------------------------------------

class HeuristicChecks(BaseModel):
    required_patterns: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    json_valid: bool = False
    min_chars: int = 0
    max_chars: int = 0
    non_empty: bool = True
    contains: list[str] = Field(default_factory=list)
    exact_match_field: Optional[str] = None  # dataset field to compare against


class JudgeConfig(BaseModel):
    provider: str = "openai"
    model: Optional[str] = None
    rubric: str = "Score the response quality from 1 to 5."
    scale: tuple[int, int] = (1, 5)
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class RagQualityConfig(BaseModel):
    context_recall_min: Optional[float] = None  # informational threshold hint


class QualityConfig(BaseModel):
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    heuristics: HeuristicChecks = Field(default_factory=HeuristicChecks)
    judge: Optional[JudgeConfig] = None
    rag: Optional[RagQualityConfig] = None


class ScenarioConfig(BaseModel):
    name: str
    target: str
    weight: float = 1.0
    think_time: Optional[tuple[float, float]] = None
    stream: Optional[bool] = None  # override target.stream
    params: dict[str, Any] = Field(default_factory=dict)  # merge over target.params
    quality: Optional[QualityConfig] = None
    tags: dict[str, str] = Field(default_factory=dict)


class LoadStage(BaseModel):
    users: int
    spawn_rate: Optional[float] = None
    duration: str  # e.g. "2m"


class LoadConfig(BaseModel):
    users: int = 1
    spawn_rate: float = 1.0
    duration: Optional[str] = None
    stages: Optional[list[LoadStage]] = None
    run_time: Optional[str] = None


class ThresholdRule(BaseModel):
    """One SLO rule. Metric names map to stats keys."""
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    max_error_rate: Optional[float] = None
    min_throughput_rps: Optional[float] = None
    min_quality_score: Optional[float] = None
    min_quality_pass_rate: Optional[float] = None
    min_context_recall: Optional[float] = None
    max_cost_usd: Optional[float] = None
    # per-stage rules: dict stage_name -> {p95_ms: ...}
    stages: dict[str, dict[str, float]] = Field(default_factory=dict)


class ThresholdsConfig(BaseModel):
    global_: Optional[ThresholdRule] = Field(default=None, alias="global")
    scenarios: dict[str, ThresholdRule] = Field(default_factory=dict)
    on_fail: Literal["fail", "warn"] = "fail"
    alerts: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class PricingConfig(BaseModel):
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    currency: str = "USD"


class ReportConfig(BaseModel):
    formats: list[Literal["csv", "json", "html"]] = Field(default_factory=lambda: ["csv", "json", "html"])
    output_dir: str = "results"
    percentiles: list[float] = Field(default_factory=lambda: [50.0, 90.0, 95.0, 99.0])
    pricing: Optional[PricingConfig] = None


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class PerfConfig(BaseModel):
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    targets: list[Target] = Field(default_factory=list)
    scenarios: list[ScenarioConfig] = Field(default_factory=list)
    load: LoadConfig = Field(default_factory=LoadConfig)
    thresholds: Optional[ThresholdsConfig] = None
    report: ReportConfig = Field(default_factory=ReportConfig)

    @model_validator(mode="after")
    def _cross_check(self):
        target_names = {t.name for t in self.targets}
        for s in self.scenarios:
            if s.target not in target_names:
                raise ValueError(f"scenario {s.name!r} references unknown target {s.target!r}")
        for name, ds in self.datasets.items():
            if not ds.source:
                raise ValueError(f"dataset {name!r} missing source")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PerfConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        # allow shorthand: top-level targets may be list of dicts w/ layer inference
        return cls.model_validate(_normalize(raw, path.parent))


def _infer_layer(t: dict) -> dict:
    """Best-effort layer inference for convenience."""
    if "layer" in t:
        return t
    if "pipeline" in t:
        t["layer"] = "pipeline"
    elif "python_path" in t:
        t["layer"] = "app_object"
    elif "framework" in t:
        t["layer"] = "app_object"
    elif t.get("provider") in ("openai", "anthropic", "azure", "bedrock", "grpc") or "params" in t:
        t["layer"] = "llm_api"
    else:
        t["layer"] = "app_endpoint"
    return t


def _normalize(raw: dict, base_dir: Path) -> dict:
    """Resolve relative paths against the config file's directory."""
    raw = dict(raw)
    for t in raw.get("targets", []) or []:
        _infer_layer(t)
    # resolve relative dataset sources and prompt/template paths
    for ds in (raw.get("datasets") or {}).values():
        src = ds.get("source", "")
        if src and not any(src.startswith(p) for p in ("http://", "https://", "s3://", "gs://")):
            p = Path(src)
            if not p.is_absolute():
                ds["source"] = str((base_dir / p).resolve())
    for t in raw.get("targets", []) or []:
        for key in ("request_template",):
            pass  # templates are file paths resolved lazily by tasks
        for st in t.get("pipeline", []) or []:
            if st.get("template"):
                p = Path(st["template"])
                if not p.is_absolute():
                    st["template"] = str((base_dir / p).resolve())
    return raw


def env_or_none(var: str) -> Optional[str]:
    return os.environ.get(var)
