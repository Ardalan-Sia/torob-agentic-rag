from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

from .tracing import TraceManager


class Monitor:
    def __init__(self, root: Path):
        self.root = root
        self.tracing = TraceManager(root)
        self.logs_dir = root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.logs_dir / "events.jsonl"
        self.metrics_path = self.logs_dir / "metrics.json"
        self.lock = Lock()
        self.interactions = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0
        self.memory_reads = 0
        self.memory_hits = 0

    def event(self, name: str, payload: dict[str, Any]) -> None:
        row = {"ts": time.time(), "event": name, "payload": payload}
        with self.tracing.span(name, payload):
            pass
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_memory_read(self, hit_count: int) -> None:
        with self.lock:
            self.memory_reads += 1
            if hit_count:
                self.memory_hits += 1
        self.event("metric.memory_read", {"hit_count": hit_count})

    def record_answer(self, latency_ms: float, tokens: int) -> None:
        with self.lock:
            self.interactions += 1
            self.total_latency_ms += latency_ms
            self.total_tokens += tokens
            self.metrics_path.write_text(
                json.dumps(self.metrics(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def metrics(self) -> dict[str, Any]:
        return {
            "interactions": self.interactions,
            "mean_time_to_answer_ms": round(
                self.total_latency_ms / self.interactions, 3
            )
            if self.interactions
            else 0.0,
            "tokens_per_message": round(self.total_tokens / self.interactions, 3)
            if self.interactions
            else 0.0,
            "memory_hit_rate": round(self.memory_hits / self.memory_reads, 3)
            if self.memory_reads
            else 0.0,
            "tracing_enabled": int(self.tracing.enabled),
            "tracing_status": self.tracing.status,
        }

    def span(self, name: str, attributes: dict[str, Any] | None = None):
        return self.tracing.span(name, attributes)


def now_ms() -> float:
    return time.perf_counter() * 1000
