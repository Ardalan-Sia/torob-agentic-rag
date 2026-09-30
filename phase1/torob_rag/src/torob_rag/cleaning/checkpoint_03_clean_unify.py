from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_config, load_all_tables
from torob_rag.utils.text import normalize_text


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    cln = raw.get("cleaning", {})
    txt_cfg = cln.get("text", {})

    price_min = int(cln.get("price_min_valid", 1))
    price_max_cap = int(cln.get("price_max_cap", 1_000_000_000))
    drop_invalid = bool(cln.get("drop_rows_with_invalid_price", True))

    out_dir = cfg.artifacts_dir / "clean"
    ensure_dir(out_dir)

    tables = load_all_tables(cfg)
    bp = tables["base_products"].copy()
    mem = tables["members"].copy()
    cats = tables["categories"].copy()
    brands = tables["brands"].copy()
    shops = tables["shops"].copy()

    # ---------- 타입/کلیدها ----------
    # base_products.random_key و members.base_random_key را string می‌کنیم برای اطمینان
    bp["random_key"] = bp["random_key"].astype(str)
    mem["base_random_key"] = mem["base_random_key"].astype(str)
    mem["random_key"] = mem["random_key"].astype(str)

    # ---------- join title ها ----------
    cats2 = cats[["id", "title"]].rename(columns={"id": "category_id", "title": "category_title"})
    brands2 = brands[["id", "title"]].rename(columns={"id": "brand_id", "title": "brand_title"})

    bp = bp.merge(cats2, on="category_id", how="left")
    bp = bp.merge(brands2, on="brand_id", how="left")

    # ---------- price cleaning ----------
    mem["price"] = pd.to_numeric(mem["price"], errors="coerce")

    before_mem = len(mem)
    invalid_mask = (mem["price"].isna()) | (mem["price"] < price_min) | (mem["price"] > price_max_cap)
    invalid_count = int(invalid_mask.sum())

    if drop_invalid:
        mem_clean = mem.loc[~invalid_mask].copy()
    else:
        mem_clean = mem.copy()
        mem_clean.loc[invalid_mask, "price"] = pd.NA

    after_mem = len(mem_clean)

    # ---------- shop join (برای aggregate کیفیت فروشگاه) ----------
    shops2 = shops[["id", "score", "has_warranty"]].rename(columns={"id": "shop_id"})
    mem_clean = mem_clean.merge(shops2, on="shop_id", how="left")

    # ---------- aggregate members -> base level ----------
    agg = mem_clean.groupby("base_random_key").agg(
        members_count=("random_key", "count"),
        shops_count=("shop_id", "nunique"),
        price_min=("price", "min"),
        price_median=("price", "median"),
        price_max=("price", "max"),
        best_shop_score=("score", "max"),
        warranty_rate=("has_warranty", "mean"),
    ).reset_index().rename(columns={"base_random_key": "random_key"})

    bp_clean = bp.merge(agg, on="random_key", how="left")

    # ---------- text field for embedding ----------
    join_fields = txt_cfg.get("join_fields", ["persian_name", "english_name", "category_title", "brand_title", "extra_features"])

    def build_text(row) -> str:
        parts = []
        for f in join_fields:
            if f in row and pd.notna(row[f]):
                parts.append(normalize_text(row[f]))
        return " | ".join([p for p in parts if p])

    bp_clean["text_for_embedding"] = bp_clean.apply(build_text, axis=1)

    # ---------- deterministic ordering ----------
    bp_clean = bp_clean.sort_values("random_key").reset_index(drop=True)
    mem_clean = mem_clean.sort_values(["base_random_key", "shop_id", "random_key"]).reset_index(drop=True)

    # ---------- save outputs ----------
    bp_out = out_dir / "base_products_clean.parquet"
    mem_out = out_dir / "members_clean.parquet"
    uni_out = out_dir / "unified_base_products.parquet"

    bp_clean.to_parquet(bp_out, index=False)
    mem_clean.to_parquet(mem_out, index=False)
    bp_clean.to_parquet(uni_out, index=False)

    # ---------- summary ----------
    summary: Dict[str, Any] = {
        "members_before": before_mem,
        "members_invalid_price": invalid_count,
        "members_after": after_mem,
        "price_min_valid": price_min,
        "price_max_cap": price_max_cap,
        "drop_invalid": drop_invalid,
        "bp_rows": int(len(bp_clean)),
        "bp_text_empty": int((bp_clean["text_for_embedding"].astype(str).str.len() == 0).sum()),
    }
    (out_dir / "cleaning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 03 cleaning pipeline generated outputs[/bold green]")
    rprint(summary)
    rprint(f"Saved: {bp_out}")
    rprint(f"Saved: {mem_out}")
    rprint(f"Saved: {uni_out}")


if __name__ == "__main__":
    main()
