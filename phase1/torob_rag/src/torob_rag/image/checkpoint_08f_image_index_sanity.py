from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import faiss
from rich import print as rprint

from torob_rag.io.load import load_config


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    run_name = "clip_v1"
    emb_path = cfg.artifacts_dir / "embeddings" / run_name / "image" / "embeddings.npy"
    index_path = cfg.artifacts_dir / "vector_db" / "faiss" / run_name / "image" / "faiss_image.index"
    out_dir = cfg.artifacts_dir / "eval" / "image" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(emb_path).astype(np.float32)
    index = faiss.read_index(str(index_path))

    top1_scores = []
    correct = 0

    bs = 256
    for i in range(0, len(X), bs):
        q = X[i:i+bs]
        scores, ids = index.search(q, 1)
        ids = ids.reshape(-1)
        scores = scores.reshape(-1)
        for j, (rid, sc) in enumerate(zip(ids, scores)):
            if rid == (i + j):
                correct += 1
            top1_scores.append(float(sc))

    acc = correct / len(X) if len(X) else 0.0
    report = {
        "run": run_name,
        "n_items": int(len(X)),
        "self_recall@1": float(acc),
        "top1_score_min": float(np.min(top1_scores)),
        "top1_score_median": float(np.median(top1_scores)),
        "top1_score_max": float(np.max(top1_scores)),
    }

    (out_dir / "sanity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rprint("[bold green]Checkpoint 08F complete[/bold green]")
    rprint(report)
    rprint(f"Saved: {out_dir / 'sanity_report.json'}")


if __name__ == "__main__":
    main()
