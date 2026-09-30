from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_image_path(img_dir: Path, rk: str) -> Path | None:
    for ext in (".jpg", ".png", ".webp"):
        p = img_dir / f"{rk}{ext}"
        if p.exists():
            return p
    return None


@torch.no_grad()
def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    run_name = raw.get("runs", {}).get("image_run", "clip_v1")
    emb_cfg = raw.get("image_embeddings", {})
    model_name = str(emb_cfg.get("model_name", "openai/clip-vit-base-patch32"))
    batch_size = int(emb_cfg.get("batch_size", 32))
    normalize = bool(emb_cfg.get("normalize", True))

    run_dir = cfg.artifacts_dir / "images" / run_name
    subset = pd.read_parquet(run_dir / "image_subset.parquet")
    img_dir = run_dir / "raw_images"

    # resolve existing images only
    rows: List[Tuple[str, str]] = []
    for rk, url in zip(subset["random_key"].astype(str), subset["image_url"].astype(str)):
        p = find_image_path(img_dir, rk)
        if p is not None:
            rows.append((rk, str(p)))

    rprint({"subset_size": int(len(subset)), "images_found": int(len(rows))})

    out_dir = cfg.artifacts_dir / "embeddings" / run_name / "image"
    ensure_dir(out_dir)

    device = torch.device("cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    embs = []
    keys = []

    for i in tqdm(range(0, len(rows), batch_size), desc="CLIP-encoding"):
        batch = rows[i:i+batch_size]
        imgs = []
        batch_keys = []
        for rk, path in batch:
            try:
                im = Image.open(path).convert("RGB")
                imgs.append(im)
                batch_keys.append(rk)
            except Exception:
                continue

        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt")
        outputs = model.get_image_features(**{k: v.to(device) for k, v in inputs.items()})
        if normalize:
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)

        embs.append(outputs.cpu().numpy().astype(np.float32))
        keys.extend(batch_keys)

    X = np.vstack(embs)
    np.save(out_dir / "embeddings.npy", X)

    idx = pd.DataFrame({"row_id": np.arange(len(keys), dtype=np.int64), "random_key": keys})
    idx.to_parquet(out_dir / "image_index.parquet", index=False)

    meta: Dict[str, Any] = {
        "run": run_name,
        "model_name": model_name,
        "dims": int(X.shape[1]),
        "n_items": int(X.shape[0]),
        "batch_size": batch_size,
        "normalize": normalize,
    }
    (out_dir / "image_embeddings_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile("./torob_rag/configs/config.yaml", out_dir / "config_used.yaml")

    rprint("[bold green]Checkpoint 08C complete[/bold green]")
    rprint(meta)
    rprint(f"Saved: {out_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
