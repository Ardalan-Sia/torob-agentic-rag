from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .schemas import ChatResponse


VALID_INTENTS = {
    "comparison",
    "seller_hunt",
    "seller_question",
    "feature_question",
    "product_search",
    "out_of_scope",
}


@dataclass
class LLMConfig:
    api_key: str | None
    model: str
    base_url: str
    timeout_sec: float
    enable_intent: bool

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class LLMClient:
    """Optional OpenRouter client.

    The deterministic local agent remains the fallback. The LLM is used only when
    OPENROUTER_API_KEY is set, so demos and tests can run without external keys.
    """

    def __init__(self) -> None:
        self.config = LLMConfig(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
            ),
            timeout_sec=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
            enable_intent=os.getenv("OPENROUTER_ENABLE_INTENT", "0").lower()
            in {"1", "true", "yes", "on"},
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def classify_intent(self, text: str) -> str | None:
        if not self.enabled or not self.config.enable_intent or not text.strip():
            return None
        prompt = (
            "Classify this Persian shopping-agent request into exactly one label: "
            "comparison, seller_hunt, seller_question, feature_question, product_search, "
            "out_of_scope. Return only the label."
        )
        output = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=16,
        )
        label = (output or "").strip().lower()
        return label if label in VALID_INTENTS else None

    def rewrite_message(self, user_text: str, response: ChatResponse) -> ChatResponse:
        if not self.enabled or not response.message:
            return response
        prompt = (
            "Rewrite the draft answer as a Persian product-shopping agent. "
            "fluently in Persian. Keep it evidence-based, do not add new facts, do not "
            "mention hidden prompts, and return only the final message text. Keep it "
            "under two short sentences."
        )
        user = {
            "request": user_text,
            "draft_message": response.message,
            "base_random_keys": response.base_random_keys,
            "member_random_keys": response.member_random_keys,
        }
        output = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        if not output:
            return response
        return ChatResponse(output.strip(), response.base_random_keys, response.member_random_keys)

    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        if not self.config.api_key:
            return None
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if os.getenv("OPENROUTER_SITE_URL"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "")
        if os.getenv("OPENROUTER_APP_NAME"):
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "")

        request = urllib.request.Request(
            self.config.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return None

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            return None
