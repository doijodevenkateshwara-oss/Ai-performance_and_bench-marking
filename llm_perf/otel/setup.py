"""OpenTelemetry setup: tracer/meter providers with OTLP export."""
from __future__ import annotations

import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

_initialized = False
_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None


def setup_otel(service_name: str = "llm-perf", endpoint: Optional[str] = None,
               sample_ratio: float = 1.0) -> None:
    """Initialize OTel SDK once per process. Safe to call multiple times."""
    global _initialized, _tracer_provider, _meter_provider
    if _initialized:
        return
    _initialized = True

    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4317"
    resource = Resource.create({
        "service.name": service_name,
        "service.namespace": "llm-perf",
        "telemetry.sdk.language": "python",
    })

    # Traces
    tp = TracerProvider(resource=resource,
                        sampler=ParentBased(TraceIdRatioBased(sample_ratio)))
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint,
                                                               insecure=True)))
    trace.set_tracer_provider(tp)
    _tracer_provider = tp

    # Metrics
    mp = MeterProvider(resource=resource, metric_readers=[
        PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint, insecure=True),
                                      export_interval_millis=15000)
    ])
    metrics.set_meter_provider(mp)
    _meter_provider = mp


def get_tracer(name: str = "llm_perf"):
    return trace.get_tracer(name)


def get_meter(name: str = "llm_perf"):
    return metrics.get_meter(name)


def shutdown() -> None:
    global _initialized
    if _tracer_provider:
        _tracer_provider.shutdown()
    if _meter_provider:
        _meter_provider.shutdown()
    _initialized = False
