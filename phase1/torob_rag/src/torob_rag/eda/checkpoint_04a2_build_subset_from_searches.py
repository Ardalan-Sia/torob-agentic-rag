from __future__ import annotations

import json
import ast
import re
from pathlib import Path
from collections import Counter
from typing import Any, Iterable, List

import pandas as pd
import pyarrow.parquet as pq
from rich import print as rprint

from torob_rag.io.load import load_config
from torob_rag.utils.text import normalize_text


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_rk_list(x: Any) -> List[str]:
    """
    result_base_product_rks can be:
      - list-like (pyarrow list)
      - stringified list
      - null
    Return list[str]
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i is not None]
    if isinstance(x, (tuple, set)):
        return [str(i) for i in x if i is not None]
    # string case
    s = str(x).strip()
    if not s:
        return []
    # try literal_eval if looks like a list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(i) for i in v if i is not None]
        except Exception:
            pass
    # fallback: split by comma
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    subcfg = raw.get("subset", {})
    max_n = int(subcfg.get("max_base_products", 200_000))

    # weights (fixed, defensible)
    W_IMP = 1
    W_VIEW = 5
    W_CLICK = 20

    out_dir = cfg.artifacts_dir / "subset"
    ensure_dir(out_dir)

    # Load base products keys (valid universe)
    base_products = pd.read_parquet(cfg.data_dir / "base_products.parquet", columns=["random_key", "persian_name", "english_name"])
    base_products["random_key"] = base_products["random_key"].astype(str)
    valid_keys = set(base_products["random_key"].tolist())

    # 1) Count impressions from searches.result_base_product_rks in batches
    searches_path = cfg.data_dir / "searches.parquet"
    pf = pq.ParquetFile(str(searches_path))

    imp_counter = Counter()
    for batch in pf.iter_batches(batch_size=5000, columns=["result_base_product_rks"]):
        col = batch.column(0)
        # convert to python objects per batch (bounded memory)
        values = col.to_pylist()
        for x in values:
            rks = parse_rk_list(x)
            for rk in rks:
                if rk in valid_keys:
                    imp_counter[rk] += 1

    rprint(f"[bold]Impression unique keys:[/bold] {len(imp_counter)}")

    # 2) Views from base_views
    base_views = pd.read_parquet(cfg.data_dir / "base_views.parquet", columns=["base_product_rk"])
    base_views["base_product_rk"] = base_views["base_product_rk"].astype(str)
    view_counts = base_views["base_product_rk"].value_counts()
    view_counter = Counter(view_counts.to_dict())
    rprint(f"[bold]View unique keys:[/bold] {len(view_counter)}")

    # 3) Final clicks -> join to base_views to get base_product_rk
    final_clicks = pd.read_parquet(cfg.data_dir / "final_clicks.parquet", columns=["base_view_id"])
    base_views_full = pd.read_parquet(cfg.data_dir / "base_views.parquet", columns=["id", "base_product_rk"])
    base_views_full["base_product_rk"] = base_views_full["base_product_rk"].astype(str)

    clicks_join = final_clicks.merge(base_views_full, left_on="base_view_id", right_on="id", how="left")
    click_counts = clicks_join["base_product_rk"].value_counts()
    click_counter = Counter(click_counts.to_dict())
    rprint(f"[bold]Final-click unique keys:[/bold] {len(click_counter)}")

    # 4) Score aggregation
    all_keys = set(imp_counter.keys()) | set(view_counter.keys()) | set(click_counter.keys())
    scored = []
    for rk in all_keys:
        if rk not in valid_keys:
            continue
        score = (
            W_IMP * imp_counter.get(rk, 0)
            + W_VIEW * view_counter.get(rk, 0)
            + W_CLICK * click_counter.get(rk, 0)
        )
        scored.append((rk, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:max_n]
    subset_keys = [rk for rk, _ in top]

    subset_df = pd.DataFrame({"random_key": subset_keys})
    subset_path = out_dir / "base_products_subset_v2.parquet"
    subset_df.to_parquet(subset_path, index=False)

    # Quick coverage checks: phone-ish keywords in names (normalized)
    bp_map = base_products.set_index("random_key")
    def name_text(rk: str) -> str:
        row = bp_map.loc[rk]
        return normalize_text(f"{row.get('persian_name','')} {row.get('english_name','')}").lower()

    phone_pat = re.compile(r"(گوشی|موبایل|تلفن همراه|smartphone|mobile phone|cell phone|phone)", re.IGNORECASE)
    samsung_pat = re.compile(r"(سامسونگ|samsung)", re.IGNORECASE)
    galaxy_pat = re.compile(r"(گلکسی|galaxy)", re.IGNORECASE)

    phone_cnt = 0
    samsung_and_phone = 0
    galaxy_cnt = 0
    # sample first N for speed check; still ok on 200k usually, but we’ll keep full (deterministic)
    for rk in subset_keys:
        t = name_text(rk)
        is_phone = bool(phone_pat.search(t))
        is_samsung = bool(samsung_pat.search(t))
        is_galaxy = bool(galaxy_pat.search(t))
        phone_cnt += int(is_phone)
        galaxy_cnt += int(is_galaxy)
        samsung_and_phone += int(is_phone and is_samsung)

    summary = {
        "subset_version": "v2_logs_weighted",
        "max_base_products": max_n,
        "subset_size": int(len(subset_keys)),
        "impression_unique": int(len(imp_counter)),
        "view_unique": int(len(view_counter)),
        "final_click_unique": int(len(click_counter)),
        "weights": {"impression": W_IMP, "view": W_VIEW, "final_click": W_CLICK},
        "coverage_in_subset": {
            "phone_keyword_in_name_count": int(phone_cnt),
            "galaxy_count": int(galaxy_cnt),
            "samsung_and_phone_kw": int(samsung_and_phone),
        },
    }

    (out_dir / "subset_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 04A-v2 complete[/bold green]")
    rprint(summary)
    rprint(f"Saved: {subset_path}")


if __name__ == "__main__":
    main()
