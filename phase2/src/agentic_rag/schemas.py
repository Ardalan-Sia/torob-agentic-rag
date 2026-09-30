from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


MessageType = Literal["text", "image"]


@dataclass
class ChatMessage:
    type: MessageType
    content: str


@dataclass
class ChatRequest:
    chat_id: str
    messages: list[ChatMessage]


@dataclass
class ChatResponse:
    message: str | None
    base_random_keys: list[str] | None
    member_random_keys: list[str] | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["base_random_keys"] is not None:
            data["base_random_keys"] = data["base_random_keys"][:10]
        if data["member_random_keys"] is not None:
            data["member_random_keys"] = data["member_random_keys"][:10]
        return data


@dataclass
class Product:
    random_key: str
    persian_name: str
    english_name: str
    category_title: str
    brand_title: str
    extra_features: str
    score: float
    price_min: float | None = None
    price_median: float | None = None
    price_max: float | None = None


@dataclass
class Member:
    random_key: str
    base_random_key: str
    price: int | float | None
    shop_id: int | str | None
    score: float | None
    has_warranty: bool | None
