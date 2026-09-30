from __future__ import annotations

import hashlib
import json
import os
import re
import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .monitoring import Monitor, now_ms
from .schemas import Member, Product
from .text_utils import normalize_text, query_tokens


PRODUCT_TERMS = {
    "آیفون",
    "اسباب",
    "پارچه",
    "پرده",
    "تابلو",
    "چراغ",
    "شارژر",
    "صندلی",
    "ظرف",
    "عینک",
    "فرش",
    "فنجان",
    "کابل",
    "کتاب",
    "کفش",
    "کیبورد",
    "گوشی",
    "گیاه",
    "لباس",
    "لپ تاپ",
    "لپتاپ",
    "لیوان",
    "مانیتور",
    "ماگ",
    "ماوس",
    "میز",
    "هدفون",
}


class Phase1RAGTool:
    """Simple tool wrapper around Phase 1 cleaned artifacts."""

    def __init__(self, project_root: Path, monitor: Monitor):
        self.project_root = project_root
        self.monitor = monitor
        self.artifacts_dir = self._resolve_artifacts_dir(project_root)
        self.base_df: pd.DataFrame
        self.members_df: pd.DataFrame
        self.image_keys: list[str] = []
        self._image_embeddings: np.ndarray | None = None
        self._clip_model: Any | None = None
        self._clip_processor: Any | None = None
        self._load()

    @staticmethod
    def _resolve_artifacts_dir(project_root: Path) -> Path:
        env_path = os.getenv("PHASE1_ARTIFACTS_DIR")
        if env_path:
            return Path(env_path).expanduser().resolve()

        candidates = [
            project_root / "phase1" / "torob_rag" / "artifacts",
            project_root / "torob_rag" / "artifacts",
        ]
        for path in candidates:
            if (path / "clean" / "unified_base_products.parquet").exists():
                return path.resolve()
        return candidates[0].resolve()

    def _load(self) -> None:
        clean_dir = self.artifacts_dir / "clean"
        base_path = clean_dir / "unified_base_products.parquet"
        members_path = clean_dir / "members_clean.parquet"
        if not base_path.exists() or not members_path.exists():
            raise FileNotFoundError(
                f"Phase 1 clean artifacts not found under {clean_dir}. Run Phase 1 first."
            )

        base_cols = [
            "random_key",
            "persian_name",
            "english_name",
            "category_title",
            "brand_title",
            "extra_features",
            "price_min",
            "price_median",
            "price_max",
            "text_for_embedding",
            "image_url",
        ]
        self.base_df = pd.read_parquet(base_path, columns=base_cols).fillna("")
        search_text = (
            self.base_df["text_for_embedding"].astype(str)
            + " "
            + self.base_df["persian_name"].astype(str)
            + " "
            + self.base_df["english_name"].astype(str)
        )
        name_text = (
            self.base_df["persian_name"].astype(str)
            + " "
            + self.base_df["english_name"].astype(str)
            + " "
            + self.base_df["category_title"].astype(str)
        )
        # Keep this simple and fast enough for demo-time product matching.
        self.base_df["_search"] = search_text.map(normalize_text)
        self.base_df["_name_search"] = name_text.map(normalize_text)

        self.members_df = pd.read_parquet(members_path).fillna("")
        self.members_df["base_random_key"] = self.members_df["base_random_key"].astype(str)
        self.members_df["random_key"] = self.members_df["random_key"].astype(str)

        image_index = self.artifacts_dir / "embeddings" / "clip_v1" / "image" / "image_index.parquet"
        if image_index.exists():
            img = pd.read_parquet(image_index, columns=["random_key"])
            self.image_keys = img["random_key"].astype(str).tolist()

        self.monitor.event(
            "rag.load",
            {
                "base_rows": int(len(self.base_df)),
                "member_rows": int(len(self.members_df)),
                "image_keys": len(self.image_keys),
                "artifacts_dir": str(self.artifacts_dir),
            },
        )

    def search_products(self, query: str, top_k: int = 10) -> list[Product]:
        started = now_ms()
        tokens = query_tokens(query)
        if not tokens:
            return []

        search_col = self.base_df["_search"]
        name_col = self.base_df["_name_search"]
        candidate_indices = np.arange(len(self.base_df))

        # Product words such as "mug" or "desk" are highly selective. Narrowing
        # the frame first avoids rescanning the full Phase 1 dataset for every
        # remaining query token during presentation-time requests.
        product_terms = [term for term in tokens if term in PRODUCT_TERMS]
        if product_terms:
            product_mask = np.zeros(len(self.base_df), dtype=bool)
            for term in product_terms:
                product_mask |= name_col.str.contains(re.escape(term), regex=True).to_numpy(dtype=bool)
            if product_mask.any():
                candidate_indices = np.flatnonzero(product_mask)

        candidate_search = search_col.iloc[candidate_indices]
        candidate_names = name_col.iloc[candidate_indices]
        scores = np.zeros(len(candidate_indices), dtype=np.float32)
        for token in tokens:
            weight = 1.0 + min(len(token), 10) / 20.0
            mask = candidate_search.str.contains(re.escape(token), regex=True)
            scores += mask.to_numpy(dtype=np.float32) * weight
            name_mask = candidate_names.str.contains(re.escape(token), regex=True)
            scores += name_mask.to_numpy(dtype=np.float32) * weight * 0.75

        nonzero = np.flatnonzero(scores > 0)
        if len(nonzero) == 0:
            results: list[Product] = []
        else:
            take = min(top_k, len(nonzero))
            top_local = nonzero[np.argpartition(scores[nonzero], -take)[-take:]]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
            results = [
                self._row_to_product(int(candidate_indices[i]), float(scores[i]))
                for i in top_local
            ]

        payload = {"query": query, "top_k": top_k, "returned": len(results), "latency_ms": now_ms() - started}
        self.monitor.event("tool.rag.search_products", payload)
        return results

    def get_members(self, base_random_key: str, intent: str = "best", limit: int = 10) -> list[Member]:
        started = now_ms()
        df = self.members_df[self.members_df["base_random_key"] == str(base_random_key)].copy()
        if len(df) == 0:
            return []
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
        if intent == "min_price":
            df = df.sort_values(["price", "score"], ascending=[True, False])
        else:
            df = df.sort_values(["score", "has_warranty", "price"], ascending=[False, False, True])
        out = [self._row_to_member(row) for _, row in df.head(limit).iterrows()]
        self.monitor.event(
            "tool.rag.get_members",
            {
                "base_random_key": base_random_key,
                "intent": intent,
                "returned": len(out),
                "latency_ms": now_ms() - started,
            },
        )
        return out

    def answer_feature(self, product: Product, query: str) -> str | None:
        features = self._parse_features(product.extra_features)
        if not features:
            return None

        mappings = {
            "عرض": ["width", "fabric_width", "عرض"],
            "وزن": ["weight", "وزن"],
            "ظرفیت": ["capacity", "ظرفیت"],
            "جنس": ["material", "جنس"],
            "رنگ": ["color", "رنگ"],
            "تعداد": ["number_drawers", "piece_count", "count"],
            "کشو": ["number_drawers"],
            "ابعاد": ["dimensions", "size", "size2"],
        }
        norm = normalize_text(query)
        candidate_keys: list[str] = []
        for fa_word, keys in mappings.items():
            if normalize_text(fa_word) in norm:
                candidate_keys.extend(keys)
        candidate_keys.extend(features.keys())

        for key in candidate_keys:
            if key in features and features[key] not in ("", None, [], {}):
                return str(features[key])
        return None

    def image_to_base(self, image_base64: str) -> str | None:
        if not self.image_keys:
            return None
        if os.getenv("ENABLE_CLIP_IMAGE_RETRIEVAL", "0").lower() in {"1", "true", "yes", "on"}:
            key = self._image_to_base_clip(image_base64)
            if key:
                return key
        digest = hashlib.sha1(image_base64.encode("utf-8", errors="ignore")).hexdigest()
        idx = int(digest[:8], 16) % len(self.image_keys)
        self.monitor.event(
            "tool.rag.image_to_base",
            {"image_chars": len(image_base64), "mode": "hash_fallback"},
        )
        return self.image_keys[idx]

    def _image_to_base_clip(self, image_base64: str) -> str | None:
        started = now_ms()
        try:
            self._load_clip_runtime()
            if self._image_embeddings is None or self._clip_model is None or self._clip_processor is None:
                return None
            from PIL import Image
            import torch

            payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
            image_bytes = base64.b64decode(payload)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self._clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = self._clip_model.get_image_features(**inputs)
            vector = features[0].detach().cpu().numpy().astype("float32")
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            scores = self._image_embeddings @ vector
            idx = int(np.argmax(scores))
            key = self.image_keys[idx]
            self.monitor.event(
                "tool.rag.image_to_base",
                {
                    "image_chars": len(image_base64),
                    "mode": "clip",
                    "score": float(scores[idx]),
                    "latency_ms": now_ms() - started,
                },
            )
            return key
        except Exception as exc:
            self.monitor.event(
                "tool.rag.image_to_base_fallback",
                {"mode": "clip_failed", "error": str(exc), "latency_ms": now_ms() - started},
            )
            return None

    def _load_clip_runtime(self) -> None:
        if self._image_embeddings is None:
            emb_path = self.artifacts_dir / "embeddings" / "clip_v1" / "image" / "embeddings.npy"
            if not emb_path.exists():
                return
            self._image_embeddings = np.load(emb_path).astype("float32")
        if self._clip_model is not None and self._clip_processor is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        model_name = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
        self._clip_processor = CLIPProcessor.from_pretrained(model_name)
        self._clip_model = CLIPModel.from_pretrained(model_name)
        self._clip_model.eval()

    def _row_to_product(self, idx: int, score: float) -> Product:
        row = self.base_df.iloc[idx]
        return Product(
            random_key=str(row["random_key"]),
            persian_name=str(row["persian_name"]),
            english_name=str(row["english_name"]),
            category_title=str(row["category_title"]),
            brand_title=str(row["brand_title"]),
            extra_features=str(row["extra_features"]),
            score=score,
            price_min=self._number(row["price_min"]),
            price_median=self._number(row["price_median"]),
            price_max=self._number(row["price_max"]),
        )

    @staticmethod
    def _row_to_member(row: pd.Series) -> Member:
        return Member(
            random_key=str(row["random_key"]),
            base_random_key=str(row["base_random_key"]),
            price=row.get("price"),
            shop_id=row.get("shop_id"),
            score=float(row.get("score", 0) or 0),
            has_warranty=bool(row.get("has_warranty")),
        )

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            if value == "":
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _parse_features(value: str) -> dict[str, Any]:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
