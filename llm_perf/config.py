"""Pydantic configuration models for llm-perf YAML configs.

Author: Venkateshwara Doijode
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class OtelConfig(BaseModel):
    """OpenTelemetry export settings (endpoint, service name, sampling)."""

    enabled: bool = True
    endpoint: str | None = None  # OTLP gRPC endpoint; falls back to env/default
    service_name: str = "llm-perf"
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    genai_semconv: bool = True


class PrometheusConfig(BaseModel):
    """Deprecated alias kept for backwards compat; OTel is the metrics path."""

    enabled: bool = False
    port: int = 9090
    interval_s: int = 15


class ObservabilityConfig(BaseModel):
    """Groups OTel and legacy Prometheus settings."""

    otel: OtelConfig = Field(default_factory=OtelConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)


class SecurityConfig(BaseModel):
    """Egress/execution guardrails for untrusted or shared configs.

    Defaults preserve historical behavior (internal services reachable,
    python tools allowed with a warning). Tighten for CI running configs
    from untrusted sources.
    """

    # when False, http(s) calls resolving to loopback/RFC1918/link-local/
    # multicast/reserved addresses are blocked (SSRF guard). Applies to
    # provider endpoints, app endpoints, agent http tools, vector-store
    # http retrieves, dataset fetches, and report scrapes.
    allow_private_networks: bool = True
    # optional host allowlist; entries match exact hosts or domain suffixes
    # ("api.example.com" also matches "x.api.example.com"; "*.example.com"
    # works too). Empty = no restriction. Applies to the same call sites.
    url_allowlist: list[str] = Field(default_factory=list)
    # gate for user-supplied Python: app_object python_path imports and
    # agent tools of type=function. "warn" allows + warns once per process,
    # "allow" is silent, "off" blocks.
    python_execution: Literal["allow", "warn", "off"] = "warn"
    # dataset fetch guards (all sources, incl. local files)
    dataset_max_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    dataset_max_rows: int = Field(default=1_000_000, gt=0)
    dataset_timeout_s: float = Field(default=30.0, gt=0)
    # cap on bytes read from agent-tool / vector-store HTTP responses
    http_max_response_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    # scrub credential-shaped strings from error stats / trace attributes
    redact: bool = True
    # set true to acknowledge the config is trusted and silence warnings
    trusted: bool = False


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetSample(BaseModel):
    """Dataset subsampling strategy (first/random/weighted + count/seed)."""

    strategy: Literal["first", "random", "weighted"] = "first"
    count: int | None = None
    seed: int = 42
    weights: dict[str, float] | None = None  # tag/column value -> weight


class DatasetConfig(BaseModel):
    """A named dataset source: location, format, column mapping, sampling."""

    source: str  # path, URL, s3://, gs://
    format: Literal["yaml", "csv", "jsonl"] | None = None  # auto-detect
    columns: dict[str, str] = Field(default_factory=dict)  # logical -> source field
    sample: DatasetSample | None = None
    vars: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _source_nonempty(cls, v: str) -> str:
        """Reject empty dataset sources."""
        if not v:
            raise ValueError("dataset source must not be empty")
        return v


# ---------------------------------------------------------------------------
# Targets / layers
# ---------------------------------------------------------------------------


class AzureConfig(BaseModel):
    """Azure OpenAI specifics: api-version and deployment field name."""

    api_version: str = "2024-10-21"
    deployment_field: str = "model"


class AwsConfig(BaseModel):
    """AWS Bedrock auth settings: region, credential style, profile."""

    region: str | None = None
    auth: Literal["env", "profile", "iam_role"] = "env"
    profile: str | None = None


class GeminiConfig(BaseModel):
    """Google Gemini/Vertex AI specifics.

    vertex=False -> AI Studio (generativelanguage.googleapis.com).
    vertex=True  -> Vertex AI. `project`+`location` select the full
    publishers/google path; omit them for express mode
    (/v1/publishers/google/models/{model}:...). `auth` picks the
    credential style: "api_key" (x-goog-api-key — express mode or AI
    Studio) or "bearer" (Authorization: Bearer — OAuth2 access token
    from `api_key_env`, e.g. `gcloud auth print-access-token`).
    """

    api_version: str = "v1beta"
    vertex: bool = False
    project: str | None = None
    location: str | None = None
    auth: Literal["api_key", "bearer"] = "api_key"


class GrpcConfig(BaseModel):
    """gRPC target settings: proto, service/method, TLS, metadata, field maps."""

    proto_path: str | None = None  # None -> bundled triton proto
    service: str = "inference.GRPCInferenceService"
    method: str = "ModelInfer"
    tls: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)
    request_fields: dict[str, str] = Field(default_factory=dict)  # logical->proto field
    response_fields: dict[str, str] = Field(default_factory=dict)


class RetryPolicy(BaseModel):
    """Bounded retry policy applied at one dependency boundary."""

    attempts: int = Field(default=0, ge=0)
    backoff_s: float = Field(default=0.5, ge=0.0)
    max_backoff_s: float = Field(default=10.0, ge=0.0)
    jitter_s: float = Field(default=0.25, ge=0.0)
    retry_on: list[
        Literal[
            "timeout",
            "connect",
            "http_4xx",
            "http_5xx",
            "llm_error",
            "empty_response",
            "exception",
        ]
    ] = Field(default_factory=lambda: ["timeout", "connect", "http_5xx"])


class CircuitBreakerPolicy(BaseModel):
    """Closed/open/half-open circuit-breaker policy for one call site."""

    enabled: bool = False
    failure_threshold: int = Field(default=3, ge=1)
    window_s: float = Field(default=30.0, gt=0.0)
    open_duration_s: float = Field(default=30.0, gt=0.0)
    half_open_calls: int = Field(default=1, ge=1)
    successes_to_close: int = Field(default=1, ge=1)


class FallbackPolicy(BaseModel):
    """Bounded fallback applied after retries fail or a circuit rejects a call."""

    type: Literal[
        "none",
        "static_response",
        "static_context",
        "empty_context",
        "passthrough",
        "alternate_llm",
        "alternate_tool",
        "alternate_endpoint",
    ] = "none"
    response: str | None = None
    context: list[str] = Field(default_factory=list)
    llm: dict[str, Any] | None = None
    tool: dict[str, Any] | None = None
    endpoint: str | None = None
    mark_degraded: bool = True


class FaultInjectionPolicy(BaseModel):
    """Deterministic/probabilistic fault window for resilience validation."""

    enabled: bool = False
    fault: Literal[
        "none",
        "timeout",
        "connect",
        "http_500",
        "http_503",
        "exception",
        "empty_response",
        "llm_error",
        "latency",
    ] = "none"
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    after_s: float = Field(default=0.0, ge=0.0)
    duration_s: float | None = Field(default=None, gt=0.0)
    latency_s: float = Field(default=0.0, ge=0.0)
    seed: int | None = None


class ResilienceConfig(BaseModel):
    """Retry, circuit-breaker, fallback, and fault-injection policies."""

    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    circuit_breaker: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)
    fallback: FallbackPolicy = Field(default_factory=FallbackPolicy)
    fault_injection: FaultInjectionPolicy = Field(default_factory=FaultInjectionPolicy)


class BaseTarget(BaseModel):
    """Fields shared by every target type (name, weight, timeouts, retries, vars)."""

    name: str
    layer: Literal["llm_api", "pipeline", "app_endpoint", "app_object"] = "llm_api"
    weight: float = 1.0  # used when referenced from scenarios; also default scenario weight
    think_time: tuple[float, float] | None = None
    timeout_s: float = 120.0
    retries: int = Field(default=0, ge=0)  # retry attempts on timeout/connect/5xx (http providers)
    retry_backoff_s: float = Field(default=0.5, ge=0.0)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    tags: dict[str, str] = Field(default_factory=dict)
    vars: dict[str, list[Any]] = Field(default_factory=dict)  # {var} template pools

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        """Validate target names (alnum plus _ and -)."""
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid target name: {v!r}")
        return v


class LlmApiTarget(BaseTarget):
    """Direct vendor-API target (OpenAI/Anthropic/Azure/Bedrock/HTTP/gRPC)."""

    layer: Literal["llm_api"] = "llm_api"
    provider: Literal["openai", "anthropic", "azure", "bedrock", "gemini", "http", "grpc"]
    transport: Literal["http", "grpc"] = "http"
    base_url: str | None = None
    endpoint: str | None = None  # http path or grpc method override
    api_key_env: str | None = None
    azure: AzureConfig | None = None
    aws: AwsConfig | None = None
    gemini: GeminiConfig | None = None
    grpc: GrpcConfig | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # model, max_tokens, ...
    stream: bool = False
    stream_mode: Literal["full", "first_token"] = "full"
    request_template: str | None = None  # jinja2 for provider=http
    response_mapping: dict[str, str] = Field(default_factory=dict)
    dataset: str | None = None
    prompts: list[str] | None = None

    @model_validator(mode="after")
    def _check_endpoint(self):
        """Apply provider default base URLs and require Azure's endpoint."""
        if self.provider in ("openai", "anthropic", "azure", "bedrock") and not self.base_url:
            # sensible defaults
            defaults = {
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com",
                "azure": None,  # must be provided
                "bedrock": None,
            }
            self.base_url = defaults.get(self.provider)
        if (
            self.transport == "http"
            and self.provider != "grpc"
            and not self.base_url
            and self.provider in ("azure",)
        ):
            raise ValueError("azure target requires base_url (https://<resource>.openai.azure.com)")
        return self


class AgentSuccessConfig(BaseModel):
    """How to decide whether an agent run succeeded."""

    expected_field: str | None = None  # dataset column matched against final answer
    match: Literal["contains", "exact", "regex"] = "contains"
    on_max_iterations: Literal["fail", "pass"] = "fail"  # does hitting max_iter count as failure?
    require_tool_use: bool = False  # success requires >=1 tool call


class PipelineStage(BaseModel):
    """One stage of a declarative pipeline: retrieve/embed/rerank/generate/agent."""

    stage: Literal["retrieve", "embed", "rerank", "generate", "agent"]
    # retrieve
    vector_store: dict[str, Any] | None = None
    top_k: int = 5
    # embed — OpenAI-compatible /embeddings endpoint (OpenAI, Ollama,
    # LM Studio, LiteLLM proxy, etc.)
    embedding: dict[str, Any] | None = None
    # rerank
    method: Literal["llm", "cross_encoder"] | None = None
    # generate / agent llm ref
    llm: dict[str, Any] | None = None
    template: str | None = None
    # agent
    max_iterations: int = 6
    max_cost_usd: float | None = None
    loop_repeat_threshold: int = Field(default=3, ge=1)  # identical tool+args calls -> loop flag
    success: AgentSuccessConfig | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    resilience: ResilienceConfig | None = None

    model_config = {"extra": "allow"}


class PipelineTarget(BaseTarget):
    """Declarative RAG/agent pipeline target bound to a dataset."""

    layer: Literal["pipeline"] = "pipeline"
    dataset: str
    pipeline: list[PipelineStage]


class AppEndpointTarget(BaseTarget):
    """Black-box application endpoint target over HTTP or gRPC."""

    layer: Literal["app_endpoint"] = "app_endpoint"
    transport: Literal["http", "grpc"] = "http"
    base_url: str
    endpoint: str | None = None
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    request_template: str | None = None
    response_mapping: dict[str, str] = Field(default_factory=dict)
    grpc: GrpcConfig | None = None
    dataset: str | None = None
    prompts: list[str] | None = None
    api_key_env: str | None = None


class AppObjectTarget(BaseTarget):
    """In-process application object target (LangChain/LlamaIndex/generic)."""

    layer: Literal["app_object"] = "app_object"
    framework: Literal["langchain", "llamaindex", "generic"] = "generic"
    python_path: str  # "module.path:factory_fn"
    call: str = "ainvoke"  # ainvoke|invoke|query|aquery|custom
    init_args: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = None
    prompts: list[str] | None = None


Target = LlmApiTarget | PipelineTarget | AppEndpointTarget | AppObjectTarget


# ---------------------------------------------------------------------------
# Scenarios / load / quality / thresholds / report
# ---------------------------------------------------------------------------


class HeuristicChecks(BaseModel):
    """Zero-cost response checks: patterns, length, JSON validity, exact match."""

    required_patterns: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    json_valid: bool = False
    min_chars: int = 0
    max_chars: int = 0
    non_empty: bool = True
    contains: list[str] = Field(default_factory=list)
    exact_match_field: str | None = None  # dataset field to compare against


class JudgeConfig(BaseModel):
    """LLM-as-judge settings: provider, model, rubric, score scale."""

    provider: str = "openai"
    model: str | None = None
    rubric: str = "Score the response quality from 1 to 5."
    scale: tuple[int, int] = (1, 5)
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RagQualityConfig(BaseModel):
    """RAG-specific quality hints (context recall)."""

    context_recall_min: float | None = None  # informational threshold hint


class QualityConfig(BaseModel):
    """Per-scenario quality evaluation: sampling, heuristics, judge, RAG."""

    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    heuristics: HeuristicChecks = Field(default_factory=HeuristicChecks)
    judge: JudgeConfig | None = None
    rag: RagQualityConfig | None = None


class ScenarioConfig(BaseModel):
    """A named load scenario binding a target to weight/think-time/quality."""

    name: str
    target: str
    weight: float = 1.0
    think_time: tuple[float, float] | None = None
    stream: bool | None = None  # override target.stream
    params: dict[str, Any] = Field(default_factory=dict)  # merge over target.params
    quality: QualityConfig | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class LoadStage(BaseModel):
    """One explicit load stage (users, spawn rate, duration) for custom profiles."""

    users: int
    spawn_rate: float | None = None
    duration: str  # e.g. "2m"


LoadProfile = Literal[
    "smoke", "baseline", "load", "stress", "spike", "soak", "scalability", "capacity", "custom"
]


class LoadConfig(BaseModel):
    """Load shape: users/spawn/duration, optional profile/stages/steps."""

    users: int = 1
    spawn_rate: float = 1.0
    duration: str | None = None
    stages: list[LoadStage] | None = None
    run_time: str | None = None
    # workload profile — when set, a LoadTestShape drives ramp/hold/down
    profile: LoadProfile | None = None
    ramp_up: str | None = None  # e.g. "2m"; defaults per profile
    ramp_down: str | None = None
    # stress / scalability / capacity step controls
    steps: int | None = None
    step_users: int | None = None
    step_duration: str | None = None
    # spike profile
    baseline_users: int | None = None
    baseline_duration: str | None = None
    # capacity profile: stop early when error rate exceeds this
    capacity_max_error_rate: float = Field(default=0.05, ge=0.0, le=1.0)

    @property
    def uses_shape(self) -> bool:
        """True when a profile or explicit stages drive a LoadTestShape."""
        return self.profile is not None or bool(self.stages)


class ThresholdRule(BaseModel):
    """One SLO rule. Metric names map to stats keys."""

    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    ttft_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    itl_p95_ms: float | None = None
    gen_p95_ms: float | None = None
    queue_p95_ms: float | None = None
    max_error_rate: float | None = None
    min_availability: float | None = None
    min_functional_availability: float | None = None
    min_non_degraded_availability: float | None = None
    max_retries: float | None = None
    max_timeouts: float | None = None
    min_throughput_rps: float | None = None
    min_quality_score: float | None = None
    min_quality_pass_rate: float | None = None
    min_context_recall: float | None = None
    # information-retrieval metrics (only evaluated when the dataset row
    # carries relevance judgments: relevant_ids / relevant_texts / expected_context)
    min_recall_at_k: float | None = None
    min_precision_at_k: float | None = None
    min_mrr: float | None = None
    min_ndcg: float | None = None
    max_cost_usd: float | None = None
    max_cost_per_success_usd: float | None = None
    # resilience rules
    max_retry_exhausted: float | None = None
    max_circuit_open_events: float | None = None
    max_circuit_rejections: float | None = None
    min_fallback_success_rate: float | None = None
    max_degraded_rate: float | None = None
    # agent-stage rules
    min_agent_success_rate: float | None = None
    max_agent_tool_error_rate: float | None = None
    max_agent_loops_avg: float | None = None
    # per-stage rules: dict stage_name -> {p95_ms: ...}
    stages: dict[str, dict[str, float]] = Field(default_factory=dict)


class ThresholdsConfig(BaseModel):
    """SLO rules (global + per-scenario), failure policy, and alerts."""

    global_: ThresholdRule | None = Field(default=None, alias="global")
    scenarios: dict[str, ThresholdRule] = Field(default_factory=dict)
    on_fail: Literal["fail", "warn"] = "fail"
    # how CI treats rules that could not be evaluated (no data):
    #   pass -> treated as pass (legacy behavior)
    #   warn -> reported as a warning, exit code unaffected
    #   fail -> nonzero exit like a real failure
    on_not_evaluated: Literal["pass", "warn", "fail"] = "pass"
    alerts: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class PricingConfig(BaseModel):
    """Per-1k-token pricing used for cost estimates."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    currency: str = "USD"


class RegressionConfig(BaseModel):
    """Baseline comparison regression detection."""

    enabled: bool = True
    on_fail: Literal["warn", "fail"] = "warn"  # fail -> nonzero exit like thresholds
    thresholds_pct: dict[str, float] = Field(default_factory=dict)  # metric -> allowed % change


class ScrapeConfig(BaseModel):
    """Scrape a Prometheus /metrics endpoint (e.g. the target under test)."""

    name: str = "target"
    url: str
    metrics: list[str] = Field(default_factory=list)  # empty = all
    timeout_s: float = 5.0


class ResourceWarnConfig(BaseModel):
    """Thresholds for load-generator health warnings in reports."""

    enabled: bool = True
    cpu_avg_pct: float = 85.0
    cpu_max_pct: float = 95.0
    mem_pct: float = 90.0
    lag_p95_ms: float = 50.0  # event-loop lag above this -> streaming metrics unreliable


class ReportConfig(BaseModel):
    """Report outputs, history/baseline/regression, resource monitoring."""

    formats: list[Literal["csv", "json", "html"]] = Field(default_factory=lambda: ["csv", "json", "html"])
    output_dir: str = "results"
    percentiles: list[float] = Field(default_factory=lambda: [50.0, 90.0, 95.0, 99.0])
    pricing: PricingConfig | None = None
    # run history + baseline comparison
    history: bool = True
    baseline: Literal["previous", "designated", "none"] | str = "previous"
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    # resource monitoring
    collect_resources: bool = True
    scrape: list[ScrapeConfig] = Field(default_factory=list)
    resource_warn: ResourceWarnConfig = Field(default_factory=ResourceWarnConfig)


class OptimizationParameter(BaseModel):
    """One bounded parameter to vary during optimization trials."""

    target: str
    stage_index: int | None = Field(default=None, ge=0)
    path: str
    values: list[Any] = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def _path_nonempty(cls, v: str) -> str:
        """Reject empty mutation paths."""
        if not v or not v.strip():
            raise ValueError("optimization parameter path must not be empty")
        return v.strip()


class OptimizationConfig(BaseModel):
    """Controlled experiment settings for benchmark-driven optimization."""

    enabled: bool = False
    scenario: str | None = None
    objective: Literal[
        "avg_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "cost_usd",
        "cost_per_success_usd",
        "agent_steps_avg",
        "agent_tool_calls_avg",
        "agent_loops_avg",
    ] = "p95_ms"
    strategy: Literal["grid", "random"] = "grid"
    max_trials: int = Field(default=10, ge=1)
    min_requests: int = Field(default=1, ge=1)
    seed: int | None = None
    constraints: ThresholdRule | None = None
    parameters: list[OptimizationParameter] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class PerfConfig(BaseModel):
    """Root configuration model for a whole test run."""

    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    targets: list[Target] = Field(default_factory=list)
    scenarios: list[ScenarioConfig] = Field(default_factory=list)
    load: LoadConfig = Field(default_factory=LoadConfig)
    thresholds: ThresholdsConfig | None = None
    report: ReportConfig = Field(default_factory=ReportConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @model_validator(mode="after")
    def _cross_check(self):
        """Cross-validate scenario->target references and dataset sources."""
        target_names = {t.name for t in self.targets}
        for s in self.scenarios:
            if s.target not in target_names:
                raise ValueError(f"scenario {s.name!r} references unknown target {s.target!r}")
        for name, ds in self.datasets.items():
            if not ds.source:
                raise ValueError(f"dataset {name!r} missing source")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PerfConfig:
        """Parse a YAML file into a validated PerfConfig."""
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
    elif "python_path" in t or "framework" in t:
        t["layer"] = "app_object"
    elif t.get("provider") in ("openai", "anthropic", "azure", "bedrock", "gemini", "grpc") or "params" in t:
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
        for st in t.get("pipeline", []) or []:
            if st.get("template"):
                p = Path(st["template"])
                if not p.is_absolute():
                    st["template"] = str((base_dir / p).resolve())
    return raw
