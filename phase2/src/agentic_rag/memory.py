from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .monitoring import Monitor
from .text_utils import contains_any, normalize_text


@dataclass
class MemoryItem:
    user_id: str
    content: str
    type: str
    created_at: float
    confidence: float = 0.75
    embedding: list[float] | None = None


@dataclass
class ChatState:
    chat_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    semantic_memory: list[MemoryItem] = field(default_factory=list)


class MemoryStore:
    def __init__(self, root: Path, monitor: Monitor, window_size: int = 8):
        self.root = root
        self.monitor = monitor
        self.window_size = window_size
        self.path = root / "runtime" / "memory.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.states: dict[str, ChatState] = {}
        self._load()

    def get(self, chat_id: str) -> ChatState:
        with self.lock:
            return self._get_unlocked(chat_id)

    def append(self, chat_id: str, role: str, content: str) -> None:
        if not content:
            return
        with self.lock:
            state = self._get_unlocked(chat_id)
            state.messages.append({"role": role, "content": content})
            state.messages = state.messages[-self.window_size :]
            self._save_unlocked()
        self.monitor.event("memory.write.short_term", {"chat_id": chat_id, "role": role})

    def update_slots(self, chat_id: str, text: str) -> None:
        updates: dict[str, str] = {}
        if contains_any(text, ["تهران", "اصفهان", "مشهد", "شیراز", "کرج", "تبریز"]):
            for city in ["تهران", "اصفهان", "مشهد", "شیراز", "کرج", "تبریز"]:
                if normalize_text(city) in normalize_text(text):
                    updates["city"] = city
                    break
        if contains_any(text, ["بودجه", "میلیون", "تومان", "ارزان", "گرون", "قیمت"]):
            updates["budget"] = text
        if contains_any(text, ["چوب", "فلز", "ام دی اف", "mdf", "ساده", "فانتزی", "کارتونی"]):
            updates["style"] = text
        if contains_any(text, ["میز", "گوشی", "آیفون", "ماگ", "پارچه", "گیاه", "لپ تاپ", "لپتاپ"]):
            updates["product_intent"] = text
        if not updates:
            return

        with self.lock:
            state = self._get_unlocked(chat_id)
            state.slots.update(updates)
            for key, value in updates.items():
                state.semantic_memory.append(
                    MemoryItem(
                        user_id=chat_id,
                        content=f"{key}: {value}",
                        type="preference" if key in {"city", "budget", "style"} else "fact",
                        created_at=time.time(),
                        embedding=self._embed_text(f"{key}: {value}"),
                    )
                )
            self._save_unlocked()
        self.monitor.event("memory.write.semantic", {"chat_id": chat_id, "keys": list(updates)})

    def retrieve(self, chat_id: str, query: str) -> list[MemoryItem]:
        state = self.get(chat_id)
        q = normalize_text(query)
        query_embedding = self._embed_text(query)
        scored: list[tuple[float, MemoryItem]] = []
        for item in state.semantic_memory:
            if item.embedding is None:
                item.embedding = self._embed_text(item.content)
            lexical_hit = any(t in q for t in normalize_text(item.content).split())
            similarity = self._cosine(query_embedding, item.embedding)
            score = similarity + (0.25 if lexical_hit else 0.0)
            if score >= 0.25:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [item for _, item in scored[:5]]
        self.monitor.record_memory_read(len(hits))
        self.monitor.event(
            "memory.read.semantic",
            {
                "chat_id": chat_id,
                "hits": len(hits),
                "scores": [round(score, 4) for score, _ in scored[:5]],
            },
        )
        return hits

    def _get_unlocked(self, chat_id: str) -> ChatState:
        if chat_id not in self.states:
            self.states[chat_id] = ChatState(chat_id=chat_id)
        return self.states[chat_id]

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for chat_id, state in raw.items():
                self.states[chat_id] = ChatState(
                    chat_id=chat_id,
                    messages=state.get("messages", []),
                    slots=state.get("slots", {}),
                    semantic_memory=[
                        MemoryItem(**item) for item in state.get("semantic_memory", [])
                    ],
                )
        except Exception as exc:
            self.monitor.event("memory.load_error", {"error": str(exc)})

    def _save_unlocked(self) -> None:
        raw = {
            chat_id: {
                "messages": state.messages,
                "slots": state.slots,
                "semantic_memory": [asdict(item) for item in state.semantic_memory],
            }
            for chat_id, state in self.states.items()
        }
        self.path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _embed_text(text: str, dims: int = 32) -> list[float]:
        vector = [0.0] * dims
        tokens = normalize_text(text).split()
        if not tokens:
            return vector
        for token in tokens:
            bucket = sum(ord(ch) for ch in token) % dims
            vector[bucket] += 1.0
            for idx, ch in enumerate(token[:6]):
                vector[(bucket + idx + ord(ch)) % dims] += 0.25
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [round(v / norm, 6) for v in vector]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        return sum(left[i] * right[i] for i in range(size))
