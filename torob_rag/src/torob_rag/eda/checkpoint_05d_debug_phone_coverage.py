from __future__ import annotations

import re
import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_config, load_all_tables


def contains_ci(s: pd.Series, pattern: str) -> pd.Series:
    return s.fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")
    tables = load_all_tables(cfg)

    uni = pd.read_parquet(cfg.artifacts_dir / "clean" / "unified_base_products.parquet")
    uni["random_key"] = uni["random_key"].astype(str)

    subset = pd.read_parquet(cfg.artifacts_dir / "subset" / "base_products_subset.parquet")
    subset_keys = set(subset["random_key"].astype(str).tolist())

    cats = tables["categories"][["id", "title", "parent_id"]].copy()
    cats["title"] = cats["title"].astype(str)

    # Join category title already exists in unified, but keep safe
    if "category_title" not in uni.columns:
        cat_map = cats.set_index("id")["title"].to_dict()
        uni["category_title"] = uni["category_id"].map(cat_map)

    # Filter to subset
    uni_s = uni[uni["random_key"].isin(subset_keys)].copy()

    rprint("[bold]Sizes[/bold]")
    rprint({"unified_total": len(uni), "subset_total": len(uni_s)})

    # Look for S23 variants in names
    name_fa = uni_s["persian_name"].fillna("").astype(str)
    name_en = uni_s["english_name"].fillna("").astype(str)

    s23_mask = contains_ci(name_fa, r"\bs\s*23\b") | contains_ci(name_en, r"\bs\s*23\b") | contains_ci(name_fa, r"\bs23\b") | contains_ci(name_en, r"\bs23\b")
    galaxy_mask = contains_ci(name_fa, r"گلکسی|galaxy") | contains_ci(name_en, r"galaxy")
    samsung_mask = contains_ci(name_fa, r"سامسونگ") | contains_ci(name_en, r"samsung")

    # Phone intent keywords inside product names (not query)
    phone_kw_mask = contains_ci(name_fa, r"گوشی|موبایل|تلفن همراه") | contains_ci(name_en, r"phone|mobile")

    rprint("\n[bold]Coverage in SUBSET[/bold]")
    rprint({
        "s23_count": int(s23_mask.sum()),
        "galaxy_count": int(galaxy_mask.sum()),
        "samsung_count": int(samsung_mask.sum()),
        "phone_keyword_in_name_count": int(phone_kw_mask.sum()),
        "samsung_and_phone_kw": int((samsung_mask & phone_kw_mask).sum()),
        "samsung_and_s23": int((samsung_mask & s23_mask).sum()),
    })

    if int(s23_mask.sum()) > 0:
        rprint("\n[bold]Sample S23 matches (up to 20)[/bold]")
        show = uni_s.loc[s23_mask, ["random_key", "persian_name", "english_name", "brand_title", "category_title"]].head(20)
        rprint(show)

    # Discover likely phone categories by title
    phone_cat_mask = cats["title"].str.contains(r"گوشی|موبایل|تلفن همراه", regex=True)
    phone_cats = cats.loc[phone_cat_mask, ["id", "title", "parent_id"]].sort_values("id")

    rprint("\n[bold]Categories with phone-ish titles[/bold]")
    rprint(phone_cats.head(50))

    phone_cat_ids = set(phone_cats["id"].tolist())

    # How many subset items belong to these categories?
    in_phone_cat = uni_s["category_id"].isin(phone_cat_ids)
    rprint("\n[bold]Subset items in discovered phone categories[/bold]")
    rprint({
        "phone_cat_ids_count": len(phone_cat_ids),
        "subset_in_phone_cats": int(in_phone_cat.sum()),
    })

    # Top categories within subset for sanity
    rprint("\n[bold]Top 15 categories in subset[/bold]")
    top_c = uni_s["category_title"].value_counts().head(15)
    rprint(top_c)

    # if phone categories exist, show top 10 sample rows
    if int(in_phone_cat.sum()) > 0:
        rprint("\n[bold]Sample products from phone categories (10)[/bold]")
        sample_phone = uni_s.loc[in_phone_cat, ["random_key", "persian_name", "brand_title", "category_title"]].head(10)
        rprint(sample_phone)


if __name__ == "__main__":
    main()
