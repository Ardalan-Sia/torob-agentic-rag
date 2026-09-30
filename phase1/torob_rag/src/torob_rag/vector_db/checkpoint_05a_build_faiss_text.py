from __future__ import annotations

from pathlib import Path
import shutil
import json

import numpy as np
import faiss
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    emb_dir = cfg.artifacts_dir / "embeddings" / "text"
    emb_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "text_embeddings_meta.json"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dims = int(meta["dims"])

    out_dir = cfg.artifacts_dir / "vector_db" / "faiss" / "text"
    ensure_dir(out_dir)

    # load embeddings (float32)
    X = np.load(emb_path)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    rprint({"loaded_embeddings": X.shape, "dtype": str(X.dtype)})

    # The embeddings are L2-normalized, so inner product equals cosine similarity.
    index = faiss.IndexFlatIP(dims)
    index.add(X)

    index_path = out_dir / "faiss_text.index"
    faiss.write_index(index, str(index_path))

    # provenance
    shutil.copyfile("./torob_rag/configs/config.yaml", out_dir / "config_used.yaml")
    shutil.copyfile(meta_path, out_dir / "text_embeddings_meta.json")

    rprint("[bold green]Checkpoint 05A complete[/bold green]")
    rprint(
        {
            "index_path": str(index_path),
            "ntotal": int(index.ntotal),
            "dims": dims,
        }
    )


if __name__ == "__main__":
    main()
