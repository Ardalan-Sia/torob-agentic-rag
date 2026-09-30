from __future__ import annotations

import ast
from pathlib import Path
import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_config


def safe_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    s = str(x)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return []


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    fail_path = cfg.artifacts_dir / "eval" / "text" / "failure_examples.csv"
    if not fail_path.exists():
        raise FileNotFoundError(f"Missing: {fail_path}")

    uni = pd.read_parquet(cfg.artifacts_dir / "clean" / "unified_base_products.parquet")
    uni["random_key"] = uni["random_key"].astype(str)
    uni_idx = uni.set_index("random_key", drop=False)

    df = pd.read_csv(fail_path)

    # Parse list columns
    df["relevant_sample_list"] = df["relevant_sample"].apply(safe_list)
    df["pred_top10_list"] = df["pred_top10"].apply(safe_list)

    def rk_to_desc(rk: str) -> str:
        rk = str(rk)
        if rk not in uni_idx.index:
            return f"{rk} | <missing>"
        row = uni_idx.loc[rk]
        return f"{rk} | {row.get('persian_name','')} | {row.get('brand_title','')} | {row.get('category_title','')}"

    # Build readable columns
    df["top1_desc"] = df["top1"].apply(rk_to_desc)
    df["relevant_desc"] = df["relevant_sample_list"].apply(lambda lst: [rk_to_desc(x) for x in lst[:5]])
    df["pred_top10_desc"] = df["pred_top10_list"].apply(lambda lst: [rk_to_desc(x) for x in lst[:10]])

    out_path = cfg.artifacts_dir / "eval" / "text" / "failure_examples_readable.csv"
    df[["query", "search_id", "num_relevant_in_index", "top1_desc", "relevant_desc", "pred_top10_desc"]].to_csv(out_path, index=False)

    rprint("[bold green]Checkpoint 07A complete[/bold green]")
    rprint(f"Saved: {out_path}")
    rprint("Sample rows:")
    rprint(df[["query", "top1_desc"]].head(5))


if __name__ == "__main__":
    main()
