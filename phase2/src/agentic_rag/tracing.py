from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class TraceManager:
    """Optional OpenTelemetry/Phoenix tracing.

    If OpenTelemetry packages or Phoenix endpoint are unavailable, this becomes a
    no-op. Local JSONL logging still records the same events through Monitor.
    """

    def __init__(self, root: Path):
        self.root = root
        self.enabled = False
        self.status = "disabled"
        self._tracer: Any = None
        self._init_otel()

    def _init_otel(self) -> None:
        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
        requested = os.getenv("ENABLE_OTEL", "0").lower() in {"1", "true", "yes", "on"}
        if not endpoint and not requested:
            return

        endpoint = endpoint or "http://127.0.0.1:6006/v1/traces"
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except Exception as exc:  # pragma: no cover - depends on optional packages
            self.status = f"otel_import_failed: {exc}"
            return

        try:
            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": "torob-agentic-rag",
                        "project.root": str(self.root),
                    }
                )
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("agentic_rag")
            self.enabled = True
            self.status = f"otel_enabled:{endpoint}"
        except Exception as exc:  # pragma: no cover - depends on local env
            self.status = f"otel_setup_failed: {exc}"

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
        if not self.enabled or self._tracer is None:
            yield
            return
        with self._tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    span.set_attribute(key, "" if value is None else value)
                else:
                    span.set_attribute(key, str(value))
            yield
