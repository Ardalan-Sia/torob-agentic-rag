from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from rich import print as rprint

from torob_rag.io.load import load_config


@torch.no_grad()
def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    mm = raw.get("multimodal", {})
    run = mm.get("run", "clip_v1")
    top_k = int(mm.get("top_k", 10))
    pool_t = int(mm.get("candidate_pool_text", 200))
    pool_i = int(mm.get("candidate_pool_image", 200))
    alpha = float(mm.get("alpha_text", 0.5))

    # load corpus for readable output
    corpus = pd.read_parquet(cfg.artifacts_dir / "multimodal" / run / "mm_corpus.parquet")
    corpus["random_key"] = corpus["random_key"].astype(str)
    corpus_idx = corpus.set_index("random_key", drop=False)

    # --- TEXT ---
    text_emb_dir = cfg.artifacts_dir / "embeddings" / run / "multimodal_text"
    t_index_map = pd.read_parquet(text_emb_dir / "text_index.parquet").sort_values("row_id")
    t_rk_arr = t_index_map["random_key"].astype(str).to_numpy()
    t_faiss = faiss.read_index(str(cfg.artifacts_dir / "vector_db" / "faiss" / run / "multimodal_text" / "faiss_text.index"))

    t_meta = json.loads((text_emb_dir / "text_embeddings_meta.json").read_text(encoding="utf-8"))
    st_model = SentenceTransformer(t_meta["model_name"])
    st_model.max_seq_length = int(t_meta["max_length"])

    # --- IMAGE ---
    i_index_map = pd.read_parquet(cfg.artifacts_dir / "embeddings" / run / "image" / "image_index.parquet").sort_values("row_id")
    i_rk_arr = i_index_map["random_key"].astype(str).to_numpy()
    i_faiss = faiss.read_index(str(cfg.artifacts_dir / "vector_db" / "faiss" / run / "image" / "faiss_image.index"))

    clip_model_name = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(clip_model_name)
    clip_proc = CLIPProcessor.from_pretrained(clip_model_name)
    clip.eval()

    # inputs
    img_path = input("Enter local image path: ").strip()
    text_query = input("Enter text query (optional): ").strip()

    # encode image
    im = Image.open(img_path).convert("RGB")
    inp = clip_proc(images=[im], return_tensors="pt")
    img_feat = clip.get_image_features(**inp)
    img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=1)
    q_img = img_feat.cpu().numpy().astype(np.float32)

    # encode text (optional)
    if text_query:
        q_text = st_model.encode([text_query], normalize_embeddings=True).astype(np.float32)
        t_scores, t_ids = t_faiss.search(q_text, pool_t)
        t_ids = t_ids[0].astype(int).tolist()
        t_scores = t_scores[0].astype(float).tolist()
        text_scores = {t_rk_arr[rid]: sc for rid, sc in zip(t_ids, t_scores)}
    else:
        text_scores = {}

    # image retrieval
    i_scores, i_ids = i_faiss.search(q_img, pool_i)
    i_ids = i_ids[0].astype(int).tolist()
    i_scores = i_scores[0].astype(float).tolist()
    image_scores = {i_rk_arr[rid]: sc for rid, sc in zip(i_ids, i_scores)}

    # fuse
    keys = set(image_scores.keys()) | set(text_scores.keys())
    fused = []
    for rk in keys:
        s_i = image_scores.get(rk, 0.0)
        s_t = text_scores.get(rk, 0.0)
        s = (alpha * s_t) + ((1 - alpha) * s_i) if text_query else s_i
        fused.append((rk, s, s_t, s_i))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_k]

    rows = []
    for rank, (rk, s, s_t, s_i) in enumerate(fused, start=1):
        row = corpus_idx.loc[rk]
        rows.append({
            "rank": rank,
            "score_fused": float(s),
            "score_text": float(s_t),
            "score_image": float(s_i),
            "persian_name": row.get("persian_name", None),
            "category_title": row.get("category_title", None),
            "price_median": row.get("price_median", None),
            "shops_count": row.get("shops_count", None),
        })

    df = pd.DataFrame(rows)
    rprint(df)


if __name__ == "__main__":
    main()
