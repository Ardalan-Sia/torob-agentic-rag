from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss
from rich import print as rprint

from torob_rag.io.load import load_config


def recall_at_k(ranked: List[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def mrr_at_k(ranked: List[str], target: str, k: int) -> float:
    for i, rk in enumerate(ranked[:k], start=1):
        if rk == target:
            return 1.0 / i
    return 0.0


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    mm = raw.get("multimodal", {})
    run = mm.get("run", "clip_v1")
    pool_t = int(mm.get("candidate_pool_text", 200))
    pool_i = int(mm.get("candidate_pool_image", 200))
    alpha = float(mm.get("alpha_text", 0.5))

    # eval settings (hard-coded sane defaults)
    sample_n = 2000
    seed = 42
    k_values = [1, 5, 10]
    max_k = max(k_values)

    # --- load text embedding + faiss ---
    t_emb_dir = cfg.artifacts_dir / "embeddings" / run / "multimodal_text"
    t_X = np.load(t_emb_dir / "embeddings.npy").astype(np.float32)
    t_map = pd.read_parquet(t_emb_dir / "text_index.parquet").sort_values("row_id")
    t_rk = t_map["random_key"].astype(str).to_numpy()
    t_rk2id = {rk: int(i) for i, rk in enumerate(t_rk)}

    t_faiss_path = cfg.artifacts_dir / "vector_db" / "faiss" / run / "multimodal_text" / "faiss_text.index"
    t_index = faiss.read_index(str(t_faiss_path))

    # --- load image embedding + faiss ---
    i_emb_dir = cfg.artifacts_dir / "embeddings" / run / "image"
    i_X = np.load(i_emb_dir / "embeddings.npy").astype(np.float32)
    i_map = pd.read_parquet(i_emb_dir / "image_index.parquet").sort_values("row_id")
    i_rk = i_map["random_key"].astype(str).to_numpy()
    i_rk2id = {rk: int(i) for i, rk in enumerate(i_rk)}

    i_faiss_path = cfg.artifacts_dir / "vector_db" / "faiss" / run / "image" / "faiss_image.index"
    i_index = faiss.read_index(str(i_faiss_path))

    # intersection keys
    keys = sorted(set(t_rk2id.keys()) & set(i_rk2id.keys()))
    if not keys:
        raise RuntimeError("No intersection between text and image keys.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(keys, size=min(sample_n, len(keys)), replace=False).tolist()

    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"mrr@{k}": [] for k in k_values})

    for rk in chosen:
        # query vectors (self)
        qt = t_X[t_rk2id[rk]].reshape(1, -1)
        qi = i_X[i_rk2id[rk]].reshape(1, -1)

        # retrieve pools
        t_scores, t_ids = t_index.search(qt, pool_t)
        i_scores, i_ids = i_index.search(qi, pool_i)

        t_ids = t_ids[0].astype(int).tolist()
        i_ids = i_ids[0].astype(int).tolist()
        t_scores = t_scores[0].astype(float).tolist()
        i_scores = i_scores[0].astype(float).tolist()

        text_scores = {t_rk[rid]: sc for rid, sc in zip(t_ids, t_scores)}
        image_scores = {i_rk[rid]: sc for rid, sc in zip(i_ids, i_scores)}

        # fuse
        cand = set(text_scores.keys()) | set(image_scores.keys())
        fused = []
        for kk in cand:
            s_t = text_scores.get(kk, 0.0)
            s_i = image_scores.get(kk, 0.0)
            s = alpha * s_t + (1 - alpha) * s_i
            fused.append((kk, s))
        fused.sort(key=lambda x: x[1], reverse=True)
        ranked = [kk for kk, _ in fused[:max_k]]

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(ranked, rk, k))
            metrics[f"mrr@{k}"].append(mrr_at_k(ranked, rk, k))

    report = {
        "run": run,
        "alpha_text": alpha,
        "pool_text": pool_t,
        "pool_image": pool_i,
        "sample_n": int(len(chosen)),
        "k_values": k_values,
        "metrics_mean": {k: float(np.mean(v)) for k, v in metrics.items()},
    }

    out_dir = cfg.artifacts_dir / "eval" / "multimodal" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mm_sanity_eval.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 09E complete[/bold green]")
    rprint(report)
    rprint(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
