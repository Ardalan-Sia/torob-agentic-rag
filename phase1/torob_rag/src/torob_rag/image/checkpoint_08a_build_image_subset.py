from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    run_name = raw.get("runs", {}).get("image_run", "clip_v1")
    max_n = int(raw.get("image_subset", {}).get("max_base_products", 20000))

    out_dir = cfg.artifacts_dir / "images" / run_name
    ensure_dir(out_dir)

    uni = pd.read_parquet(cfg.artifacts_dir / "clean" / "unified_base_products.parquet",
                          columns=["random_key", "image_url", "persian_name", "category_title", "brand_title"])
    uni["random_key"] = uni["random_key"].astype(str)

    # keep only with image_url present
    uni_img = uni[uni["image_url"].notna()].copy()

    base_views = pd.read_parquet(cfg.data_dir / "base_views.parquet", columns=["base_product_rk"])
    base_views["base_product_rk"] = base_views["base_product_rk"].astype(str)
    counts = base_views["base_product_rk"].value_counts()

    # rank products by views; keep only those with image_url
    uni_img["view_count"] = uni_img["random_key"].map(counts).fillna(0).astype(int)
    uni_img = uni_img.sort_values(["view_count", "random_key"], ascending=[False, True]).reset_index(drop=True)

    subset = uni_img.head(max_n)[["random_key", "image_url", "view_count"]].copy()
    subset_path = out_dir / "image_subset.parquet"
    subset.to_parquet(subset_path, index=False)

    summary: Dict[str, Any] = {
        "run": run_name,
        "max_base_products": max_n,
        "subset_size": int(len(subset)),
        "image_url_present_total": int(len(uni_img)),
        "top_view_count": int(subset["view_count"].max()) if len(subset) else 0,
        "p50_view_count": int(subset["view_count"].median()) if len(subset) else 0,
    }
    (out_dir / "image_subset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 08A complete[/bold green]")
    rprint(summary)
    rprint(f"Saved: {subset_path}")


if __name__ == "__main__":
    main()
