from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], batch_size: int, max_length: int, normalize: bool) -> np.ndarray:
    model.eval()
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(all_embs)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    run_name = raw.get("runs", {}).get("text_run", "e5_v1")
    e5 = raw.get("embeddings_e5", {})
    model_name = str(e5.get("model_name", "intfloat/multilingual-e5-small"))
    batch_size = int(e5.get("batch_size", 64))
    max_length = int(e5.get("max_length", 128))
    normalize = bool(e5.get("normalize", True))

    # IMPORTANT: use subset_v2 (logs-weighted)
    subset_path = cfg.artifacts_dir / "subset" / "base_products_subset_v2.parquet"
    unified_path = cfg.artifacts_dir / "clean" / "unified_base_products.parquet"

    out_dir = cfg.artifacts_dir / "embeddings" / run_name
    ensure_dir(out_dir)

    subset = pd.read_parquet(subset_path)
    subset_keys = set(subset["random_key"].astype(str).tolist())

    uni = pd.read_parquet(unified_path)
    uni["random_key"] = uni["random_key"].astype(str)
    uni = uni[uni["random_key"].isin(subset_keys)].copy().sort_values("random_key").reset_index(drop=True)

    # E5 requires "passage:" prefix
    texts = ("passage: " + uni["text_for_embedding"].astype(str)).tolist()

    rprint({"run_name": run_name, "model_name": model_name, "n_items": len(texts)})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embs = encode_texts(model, tokenizer, texts, batch_size=batch_size, max_length=max_length, normalize=normalize)

    np.save(out_dir / "embeddings.npy", embs)

    index_df = pd.DataFrame({"row_id": np.arange(len(uni), dtype=np.int64), "random_key": uni["random_key"].values})
    index_df.to_parquet(out_dir / "text_index.parquet", index=False)

    meta: Dict[str, Any] = {
        "run_name": run_name,
        "model_name": model_name,
        "dims": int(embs.shape[1]),
        "n_items": int(embs.shape[0]),
        "batch_size": batch_size,
        "max_length": max_length,
        "normalize": normalize,
        "doc_prefix": "passage:",
        "query_prefix": "query:",
    }
    (out_dir / "text_embeddings_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile("./torob_rag/configs/config.yaml", out_dir / "config_used.yaml")

    rprint("[bold green]Checkpoint 07B complete[/bold green]")
    rprint(meta)


if __name__ == "__main__":
    main()
