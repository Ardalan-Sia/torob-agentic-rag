from __future__ import annotations

from pathlib import Path

from .llm import LLMClient
from .memory import MemoryStore
from .monitoring import Monitor, now_ms
from .rag import Phase1RAGTool
from .schemas import ChatRequest, ChatResponse, Product
from .text_utils import contains_any, extract_parenthesized, normalize_text, rough_token_count


class Agent:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.phase2_root = project_root / "phase2"
        self.monitor = Monitor(self.phase2_root)
        self.memory = MemoryStore(self.phase2_root, self.monitor)
        self.rag = Phase1RAGTool(project_root, self.monitor)
        self.llm = LLMClient()
        self.monitor.event(
            "llm.setup",
            {
                "enabled": self.llm.enabled,
                "model": self.llm.config.model,
                "intent_classification": self.llm.config.enable_intent,
            },
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = now_ms()
        text = "\n".join(m.content for m in request.messages if m.type == "text").strip()
        images = [m.content for m in request.messages if m.type == "image"]
        self.memory.append(request.chat_id, "user", text or ("[image]" if images else ""))
        self.memory.update_slots(request.chat_id, text)
        self.memory.retrieve(request.chat_id, text)
        state = self.memory.get(request.chat_id)
        intent = self._resolve_intent(text, images, state.slots)
        self.monitor.event("agent.intent", {"chat_id": request.chat_id, "intent": intent})

        if intent == "injection":
            response = ChatResponse(
                "نمی‌توانم پرامپت سیستم، کانتکست پنهان یا اطلاعات داخلی را افشا کنم.",
                None,
                None,
            )
        elif intent == "image":
            response = self._image_response(images[0])
        elif intent == "empty":
            response = ChatResponse("لطفاً پیام متنی یا تصویر معتبر ارسال کنید.", None, None)
        elif intent == "comparison":
            response = self._compare(text)
        elif intent == "seller_hunt":
            response = self._seller_hunt(text, state.slots)
        elif intent == "seller_question":
            response = self._seller_question(text)
        elif intent == "feature_question":
            response = self._feature_question(text)
        elif intent == "product_search":
            response = self._product_search(text)
        else:
            response = ChatResponse(
                "اطمینان ندارم، چون این سوال خارج از محدوده داده محصولات فاز ۱ است.",
                None,
                None,
            )

        response = self.llm.rewrite_message(text, response)
        if response.message:
            self.memory.append(request.chat_id, "agent", response.message)
        self.monitor.event("agent.response", {"chat_id": request.chat_id, "response": response.to_dict()})
        self.monitor.record_answer(now_ms() - started, rough_token_count(text))
        return response

    def _resolve_intent(self, text: str, images: list[str], slots: dict) -> str:
        if self._is_injection(text):
            return "injection"
        if images:
            return "image"
        if not text:
            return "empty"

        llm_intent = self.llm.classify_intent(text)
        if llm_intent:
            return llm_intent

        if self._is_comparison(text):
            return "comparison"
        if self._is_seller_hunt(text, slots):
            return "seller_hunt"
        if self._is_seller_question(text):
            return "seller_question"
        if self._is_feature_question(text):
            return "feature_question"
        if self._is_product_question(text):
            return "product_search"
        return "out_of_scope"

    def _feature_question(self, text: str) -> ChatResponse:
        products = self.rag.search_products(text, top_k=3)
        if not products:
            return ChatResponse("محصول مشخصی در داده فاز ۱ پیدا نکردم.", None, None)
        product = products[0]
        value = self.rag.answer_feature(product, text)
        if value:
            return ChatResponse(
                f"بر اساس محصول بازیابی‌شده «{product.persian_name}»، مقدار ویژگی خواسته‌شده: {value}",
                [product.random_key],
                None,
            )
        return ChatResponse(
            f"محصول «{product.persian_name}» پیدا شد، اما مقدار این ویژگی در داده موجود نیست.",
            [product.random_key],
            None,
        )

    def _seller_question(self, text: str) -> ChatResponse:
        products = self.rag.search_products(text, top_k=3)
        if not products:
            return ChatResponse("برای پاسخ قیمت/فروشنده، نام محصول را دقیق‌تر بفرستید.", None, None)
        product = products[0]
        intent = "min_price" if contains_any(text, ["کمترین", "ارزان", "ارزون", "حداقل"]) else "best"
        members = self.rag.get_members(product.random_key, intent=intent, limit=10)
        if not members:
            return ChatResponse(
                f"محصول «{product.persian_name}» پیدا شد، اما فروشنده‌ای برای آن در داده موجود نیست.",
                [product.random_key],
                None,
            )
        selected = members[0]
        if intent == "min_price":
            msg = f"بر اساس نتایج بازیابی‌شده، کمترین قیمت برای «{product.persian_name}» برابر {selected.price} است."
        else:
            msg = f"بر اساس نتایج بازیابی‌شده، فروشنده پیشنهادی برای «{product.persian_name}» با قیمت {selected.price} است."
        return ChatResponse(msg, [product.random_key], [selected.random_key])

    def _seller_hunt(self, text: str, slots: dict) -> ChatResponse:
        has_enough_context = slots.get("product_intent") and (
            slots.get("budget") or slots.get("style") or slots.get("city")
        )
        if not has_enough_context:
            return ChatResponse(
                "برای پیدا کردن فروشنده مناسب، لطفاً بودجه، شهر، و جنس/سبک یا ابعاد مدنظرتان را بگویید.",
                None,
                None,
            )
        query = " ".join(
            str(x)
            for x in [slots.get("product_intent"), slots.get("style"), slots.get("budget"), text]
            if x
        )
        products = self.rag.search_products(query, top_k=5)
        if not products:
            return ChatResponse("با اطلاعات فعلی محصول مناسبی پیدا نکردم. نام محصول را دقیق‌تر بگویید.", None, None)
        members = self.rag.get_members(products[0].random_key, intent="best", limit=1)
        if not members:
            return ChatResponse("محصول مرتبط پیدا شد، اما فروشنده‌ای در داده موجود نیست.", None, None)
        return ChatResponse(None, None, [members[0].random_key])

    def _compare(self, text: str) -> ChatResponse:
        parts = extract_parenthesized(text)
        products: list[Product] = []
        seen: set[str] = set()
        for part in parts or [text]:
            for product in self.rag.search_products(part, top_k=2):
                if product.random_key not in seen:
                    products.append(product)
                    seen.add(product.random_key)
                break
        if not products:
            return ChatResponse("محصولات قابل مقایسه در داده فاز ۱ پیدا نشدند.", None, None)
        selected = self._choose_product(text, products)
        return ChatResponse(
            f"بر اساس نتایج بازیابی‌شده، «{selected.persian_name}» مناسب‌تر است؛ با معیارهای گفته‌شده هم‌خوانی بیشتری دارد.",
            [selected.random_key],
            None,
        )

    def _product_search(self, text: str) -> ChatResponse:
        products = self.rag.search_products(text, top_k=10)
        if not products:
            return ChatResponse("محصولی مطابق درخواست در داده فاز ۱ پیدا نکردم.", None, None)
        names = "، ".join(p.persian_name for p in products[:3])
        return ChatResponse(
            f"بر اساس نتایج بازیابی‌شده، نزدیک‌ترین گزینه‌ها: {names}",
            [p.random_key for p in products],
            None,
        )

    def _image_response(self, image: str) -> ChatResponse:
        key = self.rag.image_to_base(image)
        if not key:
            return ChatResponse("بازیابی تصویری در artifacts فاز ۱ پیدا نشد.", None, None)
        return ChatResponse(None, [key], None)

    @staticmethod
    def _choose_product(text: str, products: list[Product]) -> Product:
        if contains_any(text, ["کارتونی", "فانتزی", "کودک", "نوجوان"]):
            scored = []
            for p in products:
                doc = normalize_text(f"{p.persian_name} {p.extra_features}")
                score = sum(term in doc for term in ["کارتونی", "فانتزی", "کودک", "نوجوان"])
                scored.append((score, p.score, p))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return scored[0][2]
        return max(products, key=lambda p: p.score)

    @staticmethod
    def _is_injection(text: str) -> bool:
        return contains_any(text, ["system prompt", "پرامپت سیستم", "کانتکست پنهان", "ignore previous"])

    @staticmethod
    def _is_feature_question(text: str) -> bool:
        return contains_any(text, ["چقدر است", "چقدره", "عرض", "وزن", "ظرفیت", "جنس", "ابعاد", "رنگ", "ویژگی"]) and not Agent._is_seller_question(text)

    @staticmethod
    def _is_seller_question(text: str) -> bool:
        return contains_any(text, ["فروشنده", "فروشگاه", "کمترین قیمت", "ارزان", "ارزون", "قیمت در این پایه", "از کدوم"])

    @staticmethod
    def _is_seller_hunt(text: str, slots: dict) -> bool:
        if slots.get("product_intent") and contains_any(text, ["بودجه", "میلیون", "تومان", "چوب", "فلز", "تهران", "ساده"]):
            return True
        return contains_any(text, ["دنبال", "میخوام", "می خوام", "خرید", "بخرم", "فروشنده خوب"])

    @staticmethod
    def _is_comparison(text: str) -> bool:
        return contains_any(text, ["کدام", "کدوم", "مقایسه", "بهتر", "مناسب‌تر", "مناسب تر"]) and (
            "(" in text or contains_any(text, [" یا "])
        )

    @staticmethod
    def _is_product_question(text: str) -> bool:
        return contains_any(text, ["محصول", "گوشی", "آیفون", "ماگ", "پارچه", "گیاه", "میز", "لپ تاپ", "لپتاپ", "کفش", "لباس"])
