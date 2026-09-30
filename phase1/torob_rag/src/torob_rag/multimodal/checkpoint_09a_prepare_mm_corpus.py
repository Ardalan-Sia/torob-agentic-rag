from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    run = raw.get("multimodal", {}).get("run", "clip_v1")

    # image index keys (downloaded+embedded)
    img_idx_path = cfg.artifacts_dir / "embeddings" / run / "image" / "image_index.parquet"
    img_idx = pd.read_parquet(img_idx_path)
    keys = img_idx["random_key"].astype(str).tolist()
    key_set = set(keys)

    uni = pd.read_parquet(cfg.artifacts_dir / "clean" / "unified_base_products.parquet")
    uni["random_key"] = uni["random_key"].astype(str)
    uni = uni[uni["random_key"].isin(key_set)].copy()

    # Ensure text_for_embedding exists
    if "text_for_embedding" not in uni.columns:
        # fallback: minimal text
        uni["text_for_embedding"] = (
            uni["persian_name"].fillna("").astype(str)
            + " | " + uni["english_name"].fillna("").astype(str)
            + " | " + uni.get("brand_title", "").fillna("").astype(str)
            + " | " + uni.get("category_title", "").fillna("").astype(str)
        )

    # keep useful cols
    corpus = uni[[
        "random_key",
        "text_for_embedding",
        "persian_name",
        "english_name",
        "brand_title",
        "category_title",
        "image_url",
        "price_median",
        "shops_count",
    ]].copy()

    out_dir = cfg.artifacts_dir / "multimodal" / run
    ensure_dir(out_dir)

    out_path = out_dir / "mm_corpus.parquet"
    corpus.to_parquet(out_path, index=False)

    summary = {
        "run": run,
        "n_items": int(len(corpus)),
        "saved": str(out_path),
    }
    (out_dir / "mm_corpus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 09A complete[/bold green]")
    rprint(summary)


if __name__ == "__main__":
    main()
