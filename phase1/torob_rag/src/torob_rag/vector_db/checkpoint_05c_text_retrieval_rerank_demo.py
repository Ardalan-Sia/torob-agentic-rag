from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from rich import print as rprint

from torob_rag.io.load import load_config
from torob_rag.utils.text import normalize_text


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_query(model, tokenizer, query: str, max_length: int, normalize: bool) -> np.ndarray:
    model.eval()
    enc = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


def lexical_score(query_norm: str, doc_norm: str) -> int:
    # very simple token overlap scoring
    q_tokens = [t for t in re.split(r"\s+", query_norm) if len(t) >= 2]
    score = 0
    for t in q_tokens:
        if t in doc_norm:
            score += 1
    # extra boosts for strong intents
    if "s23" in query_norm and "s23" in doc_norm:
        score += 5
    if "گوشی" in query_norm and ("گوشی" in doc_norm or "موبایل" in doc_norm):
        score += 3
    if "iphone" in query_norm and "iphone" in doc_norm:
        score += 3
    if "galaxy" in query_norm and "galaxy" in doc_norm:
        score += 3
    return score


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    # read retrieval cfg
    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    r_cfg = raw.get("retrieval", {}).get("text", {})
    top_k = int(r_cfg.get("top_k", 10))
    candidate_pool = int(r_cfg.get("candidate_pool", 800))
    use_rerank = bool(r_cfg.get("lexical_rerank", True))

    # paths
    faiss_dir = cfg.artifacts_dir / "vector_db" / "faiss" / "text"
    index_path = faiss_dir / "faiss_text.index"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    emb_dir = cfg.artifacts_dir / "embeddings" / "text"
    meta = json.loads((emb_dir / "text_embeddings_meta.json").read_text(encoding="utf-8"))
    model_name = meta["model_name"]
    max_length = int(meta["max_length"])
    normalize = bool(meta["normalize"])

    # mapping row_id -> random_key (fast array)
    mapping = pd.read_parquet(emb_dir / "text_index.parquet").sort_values("row_id")
    rk_arr = mapping["random_key"].astype(str).to_numpy()

    # load unified base products
    uni = pd.read_parquet(cfg.artifacts_dir / "clean" / "unified_base_products.parquet")
    uni["random_key"] = uni["random_key"].astype(str)
    uni_idx = uni.set_index("random_key", drop=False)

    # load faiss
    index = faiss.read_index(str(index_path))

    # encoder
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    query = input("Enter query (fa/en): ").strip()
    if not query:
        raise ValueError("Empty query")

    # normalize query for lexical rerank
    q_norm = normalize_text(query).lower()

    q_emb = encode_query(model, tokenizer, query, max_length=max_length, normalize=normalize)

    # candidate retrieval
    scores, ids = index.search(q_emb, candidate_pool)
    ids = ids[0].astype(int)
    scores = scores[0].astype(float)

    candidates = []
    for rid, sc in zip(ids, scores):
        rk = rk_arr[rid]
        row = uni_idx.loc[rk]
        name_fa = str(row.get("persian_name", ""))
        name_en = str(row.get("english_name", ""))
        cat = str(row.get("category_title", ""))
        brand = str(row.get("brand_title", ""))

        # doc text for lexical scoring (name-focused)
        doc_norm = normalize_text(f"{name_fa} {name_en} {brand} {cat}").lower()
        lx = lexical_score(q_norm, doc_norm) if use_rerank else 0

        candidates.append(
            {
                "row_id": int(rid),
                "faiss_score": float(sc),
                "lex_score": int(lx),
                "random_key": rk,
                "persian_name": name_fa,
                "brand_title": brand,
                "category_title": cat,
                "price_median": row.get("price_median", None),
                "shops_count": row.get("shops_count", None),
            }
        )

    df = pd.DataFrame(candidates)

    if use_rerank:
        # primary: lexical, secondary: faiss
        df = df.sort_values(["lex_score", "faiss_score"], ascending=[False, False])
    else:
        df = df.sort_values(["faiss_score"], ascending=[False])

    df = df.head(top_k).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    rprint(f"\n[bold]Query:[/bold] {query}")
    rprint(df[["rank", "lex_score", "faiss_score", "persian_name", "brand_title", "category_title", "price_median", "shops_count"]])


if __name__ == "__main__":
    main()
