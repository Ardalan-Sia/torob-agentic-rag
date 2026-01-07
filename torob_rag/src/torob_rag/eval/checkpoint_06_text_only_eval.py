from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm
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
def encode_queries(model, tokenizer, queries: List[str], max_length: int, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    model.eval()
    embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        out = model(**enc)
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])
        if normalize:
            e = torch.nn.functional.normalize(e, p=2, dim=1)
        embs.append(e.cpu().numpy().astype(np.float32))
    return np.vstack(embs)


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    topk = ranked[:k]
    return 1.0 if any(x in relevant for x in topk) else 0.0


def mrr_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    topk = ranked[:k]
    for i, x in enumerate(topk, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    # binary relevance
    dcg = 0.0
    for i, x in enumerate(ranked[:k], start=1):
        rel = 1.0 if x in relevant else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)

    # ideal dcg: put all relevant first (up to k)
    m = min(len(relevant), k)
    if m == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    ev = raw.get("evaluation", {})
    sample_n = int(ev.get("sample_searches", 5000))
    k_values = list(ev.get("k_values", [1, 5, 10]))
    seed = int(ev.get("seed", 42))
    max_k = max(k_values)

    out_dir = cfg.artifacts_dir / "eval" / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load FAISS index + mapping
    faiss_index_path = cfg.artifacts_dir / "vector_db" / "faiss" / "text" / "faiss_text.index"
    index = faiss.read_index(str(faiss_index_path))

    emb_dir = cfg.artifacts_dir / "embeddings" / "text"
    meta = json.loads((emb_dir / "text_embeddings_meta.json").read_text(encoding="utf-8"))
    model_name = meta["model_name"]
    max_length = int(meta["max_length"])
    normalize = bool(meta["normalize"])

    mapping = pd.read_parquet(emb_dir / "text_index.parquet").sort_values("row_id")
    rk_arr = mapping["random_key"].astype(str).to_numpy()
    in_index = set(rk_arr.tolist())

    # load logs
    searches = pd.read_parquet(cfg.data_dir / "searches.parquet", columns=["id", "query"])
    base_views = pd.read_parquet(cfg.data_dir / "base_views.parquet", columns=["search_id", "base_product_rk"])

    # IMPORTANT: enforce same dtype for join key
    searches["id"] = searches["id"].astype(str)
    searches["query"] = searches["query"].astype(str)

    base_views["search_id"] = base_views["search_id"].astype(str)
    base_views["base_product_rk"] = base_views["base_product_rk"].astype(str)


    # group relevant base products per search_id
    rel = base_views.groupby("search_id")["base_product_rk"].apply(lambda s: set(s.tolist())).reset_index()
    rel = rel.merge(searches, left_on="search_id", right_on="id", how="left")

    # keep only searches that have a query and at least one relevant item in our index
    def has_any_in_index(s: Set[str]) -> bool:
        return any(x in in_index for x in s)

    rel = rel[rel["query"].notna()].copy()
    rel["has_in_index"] = rel["base_product_rk"].apply(has_any_in_index)
    rel = rel[rel["has_in_index"]].copy()

    rprint("[bold]Eval pool after filtering to in-index clicks:[/bold]", len(rel))

    # sample
    rel = rel.sample(n=min(sample_n, len(rel)), random_state=seed).reset_index(drop=True)

    # encode queries
    queries = rel["query"].astype(str).tolist()
    q_emb = encode_queries(AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name),
                           queries, max_length=max_length, batch_size=64, normalize=normalize)

    # search faiss
    scores, ids = index.search(q_emb.astype(np.float32), max_k)

    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"mrr@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    examples = []
    for i in tqdm(range(len(rel)), desc="Scoring"):
        relevant = set([x for x in rel.loc[i, "base_product_rk"] if x in in_index])
        ranked_ids = ids[i].astype(int).tolist()
        ranked_rks = [rk_arr[rid] for rid in ranked_ids]

        row = {
                "query": rel.loc[i, "query"],
                "search_id": rel.loc[i, "search_id"],   # keep as string
                "num_relevant_in_index": int(len(relevant)),
                "top1": ranked_rks[0],
            }

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(ranked_rks, relevant, k))
            metrics[f"mrr@{k}"].append(mrr_at_k(ranked_rks, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_rks, relevant, k))

        # store some failures for analysis
        if recall_at_k(ranked_rks, relevant, k=max_k) == 0.0:
            row["failure"] = True
            row["relevant_sample"] = list(sorted(list(relevant)))[:5]
            row["pred_top10"] = ranked_rks[:10]
            examples.append(row)

    summary = {
        "model_name": model_name,
        "index_size": int(index.ntotal),
        "eval_samples": int(len(rel)),
        "k_values": k_values,
        "metrics_mean": {k: float(np.mean(v)) for k, v in metrics.items()},
        "metrics_std": {k: float(np.std(v)) for k, v in metrics.items()},
        "failures_count": int(len(examples)),
    }

    (out_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # save a few failure examples
    ex_df = pd.DataFrame(examples[:200])
    ex_df.to_csv(out_dir / "failure_examples.csv", index=False)

    rprint("[bold green]Checkpoint 06 complete[/bold green]")
    rprint(summary)
    rprint(f"Saved: {out_dir / 'eval_summary.json'}")
    rprint(f"Saved: {out_dir / 'failure_examples.csv'}")


if __name__ == "__main__":
    main()
