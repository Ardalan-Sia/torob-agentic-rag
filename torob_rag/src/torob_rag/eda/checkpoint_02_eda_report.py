from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich import print as rprint

from torob_rag.io.load import load_config, load_all_tables


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_bar_top(series: pd.Series, title: str, out_path: Path, top_n: int = 20) -> None:
    top = series.head(top_n)
    plt.figure()
    plt.bar(range(len(top)), top.values)
    plt.xticks(range(len(top)), top.index.astype(str), rotation=75, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hist(data: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 60) -> None:
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")
    tables = load_all_tables(cfg)

    eda_cfg = (json.loads(json.dumps({})))  # placeholder to avoid mypy noise
    # read eda section from yaml by re-loading raw (simple, robust)
    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    eda_cfg = raw.get("eda", {})

    out_dir = cfg.artifacts_dir / "eda"
    ensure_dir(out_dir)

    bp = tables["base_products"]
    mem = tables["members"]
    cats = tables["categories"]
    brands = tables["brands"]
    searches = tables["searches"]
    base_views = tables["base_views"]
    final_clicks = tables["final_clicks"]

    seed = cfg.seed

    summary: Dict[str, Any] = {"tables": {}, "checks": {}}

    # ---------- basic profiles ----------
    def profile(df: pd.DataFrame, key: str | None = None) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing_by_col": (df.isna().mean().sort_values(ascending=False)).to_dict(),
        }
        if key and key in df.columns:
            p["nunique_key"] = int(df[key].nunique(dropna=True))
            p["dup_key_count"] = int(df[key].duplicated().sum())
        return p

    summary["tables"]["base_products"] = profile(bp, "random_key")
    summary["tables"]["members"] = profile(mem, "random_key")
    summary["tables"]["searches"] = profile(searches, "id")
    summary["tables"]["base_views"] = profile(base_views, "id")
    summary["tables"]["final_clicks"] = profile(final_clicks, "id")
    summary["tables"]["categories"] = profile(cats, "id")
    summary["tables"]["brands"] = profile(brands, "id")

    # ---------- price analysis ----------
    price = pd.to_numeric(mem["price"], errors="coerce")
    summary["checks"]["price"] = {
        "null": int(price.isna().sum()),
        "zero_or_neg": int((price.fillna(0) <= 0).sum()),
        "p50": float(price.quantile(0.50)),
        "p90": float(price.quantile(0.90)),
        "p95": float(price.quantile(0.95)),
        "p99": float(price.quantile(0.99)),
        "p999": float(price.quantile(0.999)),
        "max": float(price.max()),
        "min_pos": float(price[price > 0].min()) if (price > 0).any() else None,
    }

    # export top outliers
    top_out = mem.loc[price.sort_values(ascending=False).head(200).index, ["random_key", "base_random_key", "shop_id", "price"]]
    top_out.to_csv(out_dir / "price_outliers_top.csv", index=False)

    # price histogram (log10) on sample, with cap for plot stability
    n_mem_sample = int(eda_cfg.get("sample_size_members", 500_000))
    q_cap = float(eda_cfg.get("price_cap_quantile_for_plot", 0.999))
    cap_val = float(price.quantile(q_cap))

    mem_s = mem.sample(n=min(n_mem_sample, len(mem)), random_state=seed)
    price_s = pd.to_numeric(mem_s["price"], errors="coerce")
    price_s = price_s[(price_s > 0) & (price_s <= cap_val)]
    logp = np.log10(price_s.to_numpy(dtype=float))

    plot_hist(
        logp,
        title=f"Members price distribution (log10), capped at q={q_cap}",
        xlabel="log10(price)",
        out_path=out_dir / "members_price_log_hist.png",
        bins=80,
    )

    # ---------- category / brand frequency ----------
    cat_counts = bp["category_id"].value_counts()
    brand_counts = bp["brand_id"].value_counts()

    # map id -> title for nicer plots
    cats_map = cats.set_index("id")["title"].to_dict() if ("id" in cats.columns and "title" in cats.columns) else {}
    brands_map = brands.set_index("id")["title"].to_dict() if ("id" in brands.columns and "title" in brands.columns) else {}

    cat_top = cat_counts.head(int(eda_cfg.get("plots_max_categories", 20))).copy()
    cat_top.index = [cats_map.get(int(i), str(i)) for i in cat_top.index]
    plot_bar_top(cat_top, "Top categories in base_products", out_dir / "top_categories.png", top_n=len(cat_top))

    brand_top = brand_counts.head(int(eda_cfg.get("plots_max_brands", 20))).copy()
    brand_top.index = [brands_map.get(int(i), str(i)) for i in brand_top.index]
    plot_bar_top(brand_top, "Top brands in base_products", out_dir / "top_brands.png", top_n=len(brand_top))

    summary["checks"]["category_imbalance"] = {
        "num_unique_categories": int(bp["category_id"].nunique()),
        "top10_share": float(cat_counts.head(10).sum() / len(bp)),
    }
    summary["checks"]["brand_imbalance"] = {
        "num_unique_brands": int(bp["brand_id"].nunique()),
        "top10_share": float(brand_counts.head(10).sum() / len(bp)),
    }

    # ---------- members per base product distribution ----------
    per_base = mem["base_random_key"].value_counts()
    summary["checks"]["members_per_base"] = {
        "num_unique_base_in_members": int(per_base.shape[0]),
        "p50": float(per_base.quantile(0.50)),
        "p90": float(per_base.quantile(0.90)),
        "p99": float(per_base.quantile(0.99)),
        "max": int(per_base.max()),
    }

    cap = int(eda_cfg.get("members_per_base_cap", 50))
    per_base_capped = per_base.clip(upper=cap).to_numpy()
    plot_hist(
        per_base_capped,
        title=f"Members per base_product (capped at {cap})",
        xlabel="members_count (capped)",
        out_path=out_dir / "members_per_base_hist.png",
        bins=min(60, cap),
    )

    # ---------- missingness plot (base_products) ----------
    miss_bp = bp.isna().mean().sort_values(ascending=False)
    miss_bp = miss_bp[miss_bp > 0]
    if len(miss_bp) > 0:
        plt.figure()
        plt.bar(range(len(miss_bp)), miss_bp.values)
        plt.xticks(range(len(miss_bp)), miss_bp.index.astype(str), rotation=75, ha="right")
        plt.title("Missingness in base_products (only cols with >0 missing)")
        plt.tight_layout()
        plt.savefig(out_dir / "missingness_base_products.png", dpi=150)
        plt.close()
    else:
        (out_dir / "missingness_base_products.png").write_text("No missingness in base_products", encoding="utf-8")

    # ---------- search query analysis ----------
    n_search_sample = int(eda_cfg.get("sample_size_searches", 200_000))
    s_s = searches.sample(n=min(n_search_sample, len(searches)), random_state=seed)

    q = s_s["query"].astype(str)
    qlen = q.str.len().to_numpy()
    summary["checks"]["search_query"] = {
        "sample_size": int(len(s_s)),
        "len_p50": float(np.quantile(qlen, 0.50)),
        "len_p95": float(np.quantile(qlen, 0.95)),
        "len_p99": float(np.quantile(qlen, 0.99)),
    }
    plot_hist(qlen, "Search query length (sample)", "len(query)", out_dir / "search_query_len_hist.png", bins=60)

    # ---------- timestamp ranges ----------
    def ts_range(df: pd.DataFrame, col: str) -> Dict[str, Any]:
        if col not in df.columns:
            return {}
        ts = pd.to_datetime(df[col], errors="coerce", utc=True)
        return {
            "null": int(ts.isna().sum()),
            "min": None if ts.notna().sum() == 0 else str(ts.min()),
            "max": None if ts.notna().sum() == 0 else str(ts.max()),
        }

    summary["checks"]["timestamps"] = {
        "searches": ts_range(searches, "timestamp"),
        "base_views": ts_range(base_views, "timestamp"),
        "final_clicks": ts_range(final_clicks, "timestamp"),
    }

    # ---------- write artifacts ----------
    save_json(summary, out_dir / "summary.json")

    # Markdown report
    report = []
    report.append("# Torob Dataset EDA Report (Checkpoint 02)\n")
    report.append("## Key Findings (auto-generated)\n")
    report.append(f"- members price max: {summary['checks']['price']['max']}\n")
    report.append(f"- members price p99.9: {summary['checks']['price']['p999']}\n")
    report.append(f"- members price zero/neg: {summary['checks']['price']['zero_or_neg']}\n")
    report.append(f"- members_per_base max: {summary['checks']['members_per_base']['max']}\n")
    report.append(f"- category top10 share: {summary['checks']['category_imbalance']['top10_share']:.4f}\n")
    report.append(f"- brand top10 share: {summary['checks']['brand_imbalance']['top10_share']:.4f}\n\n")

    report.append("## Plots\n")
    report.append("### Price (log10, capped)\n")
    report.append("![](members_price_log_hist.png)\n\n")
    report.append("### Members per base_product (capped)\n")
    report.append("![](members_per_base_hist.png)\n\n")
    report.append("### Top categories\n")
    report.append("![](top_categories.png)\n\n")
    report.append("### Top brands\n")
    report.append("![](top_brands.png)\n\n")
    report.append("### Missingness (base_products)\n")
    report.append("![](missingness_base_products.png)\n\n")
    report.append("### Search query length\n")
    report.append("![](search_query_len_hist.png)\n\n")

    (out_dir / "EDA_REPORT.md").write_text("".join(report), encoding="utf-8")

    rprint("[bold green]Checkpoint 02 complete[/bold green]")
    rprint(f"Artifacts saved in: {out_dir}")

    # print a concise console summary for you to paste back here
    rprint("[bold]Key console summary:[/bold]")
    rprint(f"- price max: {summary['checks']['price']['max']}")
    rprint(f"- price p99.9: {summary['checks']['price']['p999']}")
    rprint(f"- count(price<=0): {summary['checks']['price']['zero_or_neg']}")
    rprint(f"- members_per_base max: {summary['checks']['members_per_base']['max']}")
    rprint(f"- search query len p95: {summary['checks']['search_query']['len_p95']}")


if __name__ == "__main__":
    main()
