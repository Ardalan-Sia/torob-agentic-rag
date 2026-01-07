from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import faiss
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    emb_dir = cfg.artifacts_dir / "embeddings" / "text"
    emb_path = emb_dir / "embeddings.npy"
    idx_path = emb_dir / "text_index.parquet"
    meta_path = emb_dir / "text_embeddings_meta.json"

    clean_uni_path = cfg.artifacts_dir / "clean" / "unified_base_products.parquet"

    out_dir = cfg.artifacts_dir / "faiss" / "text"
    ensure_dir(out_dir)

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing text_index: {idx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    d = int(meta["dims"])

    # memory-map embeddings
    x = np.load(str(emb_path), mmap_mode="r")  # (N, d) float32
    if x.dtype != np.float32:
        raise ValueError("embeddings must be float32")
    if x.shape[1] != d:
        raise ValueError(f"dims mismatch: meta={d} arr={x.shape[1]}")

    rprint({"embeddings_shape": x.shape, "dims": d})

    # Build Flat IP index (cosine if normalized)
    index = faiss.IndexFlatIP(d)

    batch = 50000
    for i in range(0, x.shape[0], batch):
        xb = np.ascontiguousarray(x[i:i + batch])
        index.add(xb)

    faiss_path = out_dir / "faiss_text.index"
    faiss.write_index(index, str(faiss_path))

    # Build a small catalog parquet for only these 200k items (fast demo)
    text_index = pd.read_parquet(idx_path)
    keys = text_index["random_key"].astype(str).tolist()
    key_set = set(keys)

    cols = [
        "random_key",
        "persian_name",
        "english_name",
        "category_id",
        "brand_id",
        "category_title",
        "brand_title",
        "image_url",
        "members_count",
        "shops_count",
        "price_min",
        "price_median",
        "price_max",
    ]
    uni = pd.read_parquet(clean_uni_path, columns=[c for c in cols if c in pd.read_parquet(clean_uni_path, nrows=1).columns])
    uni["random_key"] = uni["random_key"].astype(str)
    uni = uni.loc[uni["random_key"].isin(key_set)].copy()
    uni = uni.sort_values("random_key").reset_index(drop=True)

    catalog_path = out_dir / "catalog_subset.parquet"
    uni.to_parquet(catalog_path, index=False)

    # meta
    out_meta: Dict[str, Any] = {
        "faiss_type": "IndexFlatIP",
        "dims": d,
        "n_items": int(x.shape[0]),
        "normalized_embeddings": bool(meta.get("normalize", True)),
        "source_embeddings": str(emb_path),
        "source_index": str(idx_path),
        "catalog_subset": str(catalog_path),
        "model_name": meta.get("model_name"),
    }
    (out_dir / "faiss_text_meta.json").write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copyfile("./torob_rag/configs/config.yaml", out_dir / "config_used.yaml")

    rprint("[bold green]Checkpoint 05A complete[/bold green]")
    rprint({"faiss_index": str(faiss_path), "catalog_subset": str(catalog_path), "meta": out_meta})


if __name__ == "__main__":
    main()
