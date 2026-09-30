from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import faiss
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    run_name = raw.get("runs", {}).get("image_run", "clip_v1")

    emb_dir = cfg.artifacts_dir / "embeddings" / run_name / "image"
    X = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    meta = json.loads((emb_dir / "image_embeddings_meta.json").read_text(encoding="utf-8"))
    dims = int(meta["dims"])

    out_dir = cfg.artifacts_dir / "vector_db" / "faiss" / run_name / "image"
    ensure_dir(out_dir)

    index = faiss.IndexFlatIP(dims)
    index.add(X)

    index_path = out_dir / "faiss_image.index"
    faiss.write_index(index, str(index_path))

    rprint("[bold green]Checkpoint 08D complete[/bold green]")
    rprint({"index_path": str(index_path), "ntotal": int(index.ntotal), "dims": dims})


if __name__ == "__main__":
    main()
