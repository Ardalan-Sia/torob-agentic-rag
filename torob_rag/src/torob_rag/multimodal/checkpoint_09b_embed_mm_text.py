from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    mm = raw.get("multimodal", {})
    run = mm.get("run", "clip_v1")
    model_name = mm.get("text_model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    batch_size = int(mm.get("text_batch_size", 64))
    max_length = int(mm.get("text_max_length", 128))
    normalize = bool(mm.get("normalize", True))

    corpus_path = cfg.artifacts_dir / "multimodal" / run / "mm_corpus.parquet"
    corpus = pd.read_parquet(corpus_path)
    corpus["random_key"] = corpus["random_key"].astype(str)

    out_dir = cfg.artifacts_dir / "embeddings" / run / "multimodal_text"
    ensure_dir(out_dir)

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_length

    texts = corpus["text_for_embedding"].fillna("").astype(str).tolist()

    # encode in batches
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)

    np.save(out_dir / "embeddings.npy", embs)

    idx = pd.DataFrame({"row_id": np.arange(len(corpus), dtype=np.int64), "random_key": corpus["random_key"].values})
    idx.to_parquet(out_dir / "text_index.parquet", index=False)

    meta: Dict[str, Any] = {
        "run": run,
        "model_name": model_name,
        "dims": int(embs.shape[1]),
        "n_items": int(embs.shape[0]),
        "batch_size": batch_size,
        "max_length": max_length,
        "normalize": normalize,
    }
    (out_dir / "text_embeddings_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile("./torob_rag/configs/config.yaml", out_dir / "config_used.yaml")

    rprint("[bold green]Checkpoint 09B complete[/bold green]")
    rprint(meta)


if __name__ == "__main__":
    main()
