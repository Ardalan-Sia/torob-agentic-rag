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
    run = raw.get("multimodal", {}).get("run", "clip_v1")

    emb_dir = cfg.artifacts_dir / "embeddings" / run / "multimodal_text"
    X = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    meta = json.loads((emb_dir / "text_embeddings_meta.json").read_text(encoding="utf-8"))
    dims = int(meta["dims"])

    out_dir = cfg.artifacts_dir / "vector_db" / "faiss" / run / "multimodal_text"
    ensure_dir(out_dir)

    index = faiss.IndexFlatIP(dims)
    index.add(X)

    out_path = out_dir / "faiss_text.index"
    faiss.write_index(index, str(out_path))

    rprint("[bold green]Checkpoint 09C complete[/bold green]")
    rprint({"index_path": str(out_path), "ntotal": int(index.ntotal), "dims": dims})


if __name__ == "__main__":
    main()
