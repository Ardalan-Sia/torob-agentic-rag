from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rich import print as rprint

from torob_rag.io.load import load_config


@torch.no_grad()
def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    run_name = raw.get("runs", {}).get("image_run", "clip_v1")
    emb_cfg = raw.get("image_embeddings", {})
    model_name = str(emb_cfg.get("model_name", "openai/clip-vit-base-patch32"))
    normalize = bool(emb_cfg.get("normalize", True))

    # load index + mapping
    idx_path = cfg.artifacts_dir / "embeddings" / run_name / "image" / "image_index.parquet"
    mapping = pd.read_parquet(idx_path).sort_values("row_id")
    rk_arr = mapping["random_key"].astype(str).to_numpy()

    faiss_path = cfg.artifacts_dir / "vector_db" / "faiss" / run_name / "image" / "faiss_image.index"
    index = faiss.read_index(str(faiss_path))

    uni = pd.read_parquet(cfg.artifacts_dir / "clean" / "unified_base_products.parquet",
                          columns=["random_key","persian_name","category_title","brand_title","image_url","price_median","shops_count"])
    uni["random_key"] = uni["random_key"].astype(str)
    uni_idx = uni.set_index("random_key", drop=False)

    # encoder
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    img_path = input("Enter local image path: ").strip()
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(p)

    im = Image.open(p).convert("RGB")
    inputs = processor(images=[im], return_tensors="pt")
    feat = model.get_image_features(**inputs)
    if normalize:
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    q = feat.cpu().numpy().astype(np.float32)

    top_k = 10
    scores, ids = index.search(q, top_k)
    ids = ids[0].astype(int).tolist()
    scores = scores[0].tolist()

    results = []
    for rank, (rid, sc) in enumerate(zip(ids, scores), start=1):
        rk = rk_arr[rid]
        row = uni_idx.loc[rk]
        results.append({
            "rank": rank,
            "score": float(sc),
            "persian_name": row.get("persian_name", None),
            "brand_title": row.get("brand_title", None),
            "category_title": row.get("category_title", None),
            "price_median": row.get("price_median", None),
            "shops_count": row.get("shops_count", None),
            "image_url": row.get("image_url", None),
            "random_key": rk,
        })

    df = pd.DataFrame(results)
    rprint(df[["rank","score","persian_name","category_title","price_median","shops_count"]])


if __name__ == "__main__":
    main()
