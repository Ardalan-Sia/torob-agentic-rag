from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Set

import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_config, load_all_tables


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    subcfg = raw.get("subset", {})
    max_n = int(subcfg.get("max_base_products", 200_000))
    fill_strategy = str(subcfg.get("fill_strategy", "stratified_by_category"))
    per_cat_cap = int(subcfg.get("per_category_cap", 3000))

    out_dir = cfg.artifacts_dir / "subset"
    ensure_dir(out_dir)

    tables = load_all_tables(cfg)
    bp = tables["base_products"].copy()
    base_views = tables["base_views"].copy()

    # base product keys as str
    bp["random_key"] = bp["random_key"].astype(str)
    base_views["base_product_rk"] = base_views["base_product_rk"].astype(str)

    clicked_keys: Set[str] = set(base_views["base_product_rk"].dropna().unique())
    clicked_keys = clicked_keys.intersection(set(bp["random_key"].unique()))

    clicked_count = len(clicked_keys)
    rprint(f"[bold]Unique clicked base products:[/bold] {clicked_count}")

    # Start subset with clicked products
    subset_keys = list(clicked_keys)

    # Fill if needed
    if len(subset_keys) < max_n and fill_strategy == "stratified_by_category":
        remaining = max_n - len(subset_keys)
        rprint(f"Filling subset with stratified sampling by category. Need: {remaining}")

        bp_left = bp.loc[~bp["random_key"].isin(subset_keys), ["random_key", "category_id"]].copy()

        # deterministic: sort then group sample with random_state
        bp_left = bp_left.sort_values(["category_id", "random_key"]).reset_index(drop=True)

        add_keys = []
        for cat_id, g in bp_left.groupby("category_id", sort=True):
            if len(add_keys) >= remaining:
                break
            take = min(per_cat_cap, remaining - len(add_keys), len(g))
            # deterministic take from top (already sorted)
            add_keys.extend(g["random_key"].head(take).tolist())

        subset_keys.extend(add_keys)

    # Final trim and deterministic order
    subset_keys = sorted(subset_keys)[:max_n]

    subset_df = pd.DataFrame({"random_key": subset_keys})
    subset_path = out_dir / "base_products_subset.parquet"
    subset_df.to_parquet(subset_path, index=False)

    summary: Dict[str, Any] = {
        "clicked_unique_base_products": clicked_count,
        "subset_size": int(len(subset_keys)),
        "max_base_products": max_n,
        "fill_strategy": fill_strategy,
        "per_category_cap": per_cat_cap,
    }
    (out_dir / "subset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 04A complete[/bold green]")
    rprint(summary)
    rprint(f"Saved subset: {subset_path}")


if __name__ == "__main__":
    main()
