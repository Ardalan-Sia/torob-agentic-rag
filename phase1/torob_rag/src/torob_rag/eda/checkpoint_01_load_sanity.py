from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_all_tables, load_config


def _basic_profile(df: pd.DataFrame, key_candidates: list[str]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rate": float(df.isna().mean().mean()) if df.shape[1] > 0 else 0.0,
    }

    # key-like uniqueness checks
    uniq: Dict[str, Any] = {}
    for c in key_candidates:
        if c in df.columns:
            nunique = int(df[c].nunique(dropna=True))
            uniq[c] = {"nunique": nunique, "uniq_rate": float(nunique / max(len(df), 1))}
    profile["key_uniqueness"] = uniq
    return profile


def _consistency_checks(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    bp = tables["base_products"]
    mem = tables["members"]
    shops = tables["shops"]
    cats = tables["categories"]
    brands = tables["brands"]

    checks: Dict[str, Any] = {}

    # FK: members.base_random_key ⊆ base_products.random_key
    if "base_random_key" in mem.columns and "random_key" in bp.columns:
        bp_keys = set(bp["random_key"].dropna().astype(str).unique())
        mem_fk = mem["base_random_key"].dropna().astype(str)
        missing_fk = int((~mem_fk.isin(bp_keys)).sum())
        checks["members_base_fk_missing"] = missing_fk

    # FK: members.shop_id ⊆ shops.id
    if "shop_id" in mem.columns and "id" in shops.columns:
        shop_ids = set(shops["id"].dropna().unique())
        missing_shop = int((~mem["shop_id"].dropna().isin(shop_ids)).sum())
        checks["members_shop_fk_missing"] = missing_shop

    # FK: base_products.category_id ⊆ categories.id
    if "category_id" in bp.columns and "id" in cats.columns:
        cat_ids = set(cats["id"].dropna().unique())
        missing_cat = int((~bp["category_id"].dropna().isin(cat_ids)).sum())
        checks["base_products_category_fk_missing"] = missing_cat

    # FK: base_products.brand_id ⊆ brands.id
    if "brand_id" in bp.columns and "id" in brands.columns:
        brand_ids = set(brands["id"].dropna().unique())
        missing_brand = int((~bp["brand_id"].dropna().isin(brand_ids)).sum())
        checks["base_products_brand_fk_missing"] = missing_brand

    # Price quick scan
    if "price" in mem.columns:
        price = pd.to_numeric(mem["price"], errors="coerce")
        checks["members_price_null"] = int(price.isna().sum())
        checks["members_price_zero_or_neg"] = int((price.fillna(0) <= 0).sum())
        if price.notna().any():
            checks["members_price_p50"] = float(price.quantile(0.50))
            checks["members_price_p95"] = float(price.quantile(0.95))
            checks["members_price_p99"] = float(price.quantile(0.99))
            checks["members_price_max"] = float(price.max())

    return checks


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    tables = load_all_tables(cfg)

    # profiles
    profiles: Dict[str, Any] = {}
    key_candidates = ["id", "random_key", "uid", "search_id", "base_product_rk", "base_random_key", "shop_id"]
    for name, df in tables.items():
        profiles[name] = _basic_profile(df, key_candidates)

    checks = _consistency_checks(tables)

    out = {"profiles": profiles, "consistency": checks}
    out_path = cfg.artifacts_dir / "checkpoint_01_sanity.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 01 complete[/bold green]")
    rprint(f"Saved: {out_path}")

    # concise console summary
    for tname in ["base_products", "members", "searches", "base_views", "final_clicks"]:
        p = profiles.get(tname, {})
        rprint(f"[bold]{tname}[/bold] rows={p.get('rows')} cols={p.get('cols')} missing_rate={p.get('missing_rate'):.4f}")

    rprint("[bold]Consistency checks:[/bold]")
    for k, v in checks.items():
        rprint(f"- {k}: {v}")


if __name__ == "__main__":
    main()
