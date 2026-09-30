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
    # last_hidden: [B, T, H], mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                  # [B, 1]
    return summed / counts


@torch.no_grad()
def encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    normalize: bool,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])  # [B, H]

        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embs.append(emb.cpu().numpy().astype(np.float32))

    return np.vstack(all_embs)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    # read raw yaml for embedding cfg + provenance
    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    emb_cfg = raw.get("embeddings", {}).get("text", {})
    model_name = str(emb_cfg.get("model_name", "intfloat/multilingual-e5-small"))
    batch_size = int(emb_cfg.get("batch_size", 64))
    max_length = int(emb_cfg.get("max_length", 128))
    normalize = bool(emb_cfg.get("normalize", True))

    # paths
    clean_dir = cfg.artifacts_dir / "clean"
    subset_dir = cfg.artifacts_dir / "subset"
    out_dir = cfg.artifacts_dir / "embeddings" / "text"
    ensure_dir(out_dir)

    unified_path = clean_dir / "unified_base_products.parquet"
    subset_path = subset_dir / "base_products_subset.parquet"
    if not unified_path.exists():
        raise FileNotFoundError(f"Missing cleaned unified parquet: {unified_path}")
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing subset parquet: {subset_path}")

    # load subset keys
    subset = pd.read_parquet(subset_path)
    subset_keys = subset["random_key"].astype(str)
    subset_set = set(subset_keys.tolist())

    # load unified, filter to subset
    uni = pd.read_parquet(unified_path)
    uni["random_key"] = uni["random_key"].astype(str)
    uni = uni.loc[uni["random_key"].isin(subset_set)].copy()

    # deterministic ordering
    uni = uni.sort_values("random_key").reset_index(drop=True)

    # build texts with E5 prefixes
    # E5 expects "passage: ..." for documents
    texts = ("passage: " + uni["text_for_embedding"].astype(str)).tolist()

    rprint("[bold]Embedding config:[/bold]")
    rprint(
        {
            "model_name": model_name,
            "batch_size": batch_size,
            "max_length": max_length,
            "normalize": normalize,
            "n_items": len(texts),
        }
    )

    # device CPU
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
        device=device,
    )

    # sanity checks
    if np.isnan(embs).any() or np.isinf(embs).any():
        raise ValueError("Embeddings contain NaN or Inf!")

    dims = int(embs.shape[1])
    n = int(embs.shape[0])

    # save embeddings
    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embs)

    # save index mapping
    index_df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "random_key": uni["random_key"].astype(str).values,
        }
    )
    index_path = out_dir / "text_index.parquet"
    index_df.to_parquet(index_path, index=False)

    # meta + provenance
    meta: Dict[str, Any] = {
        "model_name": model_name,
        "dims": dims,
        "n_items": n,
        "batch_size": batch_size,
        "max_length": max_length,
        "normalize": normalize,
    }
    (out_dir / "text_embeddings_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # copy config for provenance
    shutil.copyfile("./torob_rag/configs/config.yaml", out_dir / "config_used.yaml")

    rprint("[bold green]Checkpoint 04B complete[/bold green]")
    rprint(
        {
            "embeddings_shape": embs.shape,
            "saved_embeddings": str(emb_path),
            "saved_index": str(index_path),
            "meta": meta,
        }
    )


if __name__ == "__main__":
    main()
