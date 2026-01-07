from __future__ import annotations

from pathlib import Path
import pandas as pd
from rich import print as rprint

from torob_rag.io.load import load_config


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    img_cfg = raw.get("images", {})
    images_dir = Path(img_cfg["images_dir"]).expanduser().resolve()
    allowed_ext = set([e.lower() for e in img_cfg.get("allowed_ext", [".jpg", ".jpeg", ".png", ".webp"])])
    mode = img_cfg.get("filename_mode", "stem_is_random_key")

    subset_path = cfg.artifacts_dir / "subset" / "base_products_subset.parquet"
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing subset: {subset_path}")
    subset = pd.read_parquet(subset_path)
    subset_keys = set(subset["random_key"].astype(str).tolist())

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    rows = []
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed_ext:
            continue

        if mode == "stem_is_random_key":
            rk = p.stem.strip()
        else:
            # placeholder for other naming modes
            rk = None

        if rk and rk in subset_keys:
            rows.append({"random_key": rk, "image_path": str(p)})

    out_dir = cfg.artifacts_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).drop_duplicates(subset=["random_key"]).sort_values("random_key").reset_index(drop=True)
    out_path = out_dir / "image_manifest.parquet"
    df.to_parquet(out_path, index=False)

    rprint("[bold green]Checkpoint 04C complete[/bold green]")
    rprint({"images_found_in_subset": int(len(df)), "saved": str(out_path)})


if __name__ == "__main__":
    main()
