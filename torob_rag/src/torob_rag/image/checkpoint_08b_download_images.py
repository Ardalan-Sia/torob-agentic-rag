from __future__ import annotations

import os
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple

import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
from rich import print as rprint

from torob_rag.io.load import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def guess_ext(content_type: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    return ".jpg"


def download_one(session: requests.Session, rk: str, url: str, out_dir: Path, timeout: int, max_retries: int, ua: str) -> Tuple[str, bool, str]:
    # if exists (any ext), skip
    for ext in (".jpg", ".png", ".webp"):
        if (out_dir / f"{rk}{ext}").exists():
            return rk, True, "cached"

    headers = {"User-Agent": ua}
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout, stream=True)
            if resp.status_code != 200:
                last_err = f"http_{resp.status_code}"
                time.sleep(0.2)
                continue

            ext = guess_ext(resp.headers.get("Content-Type", ""))
            tmp = out_dir / f"{rk}{ext}.tmp"
            final = out_dir / f"{rk}{ext}"

            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)

            # verify image
            try:
                with Image.open(tmp) as im:
                    im.verify()
            except Exception:
                tmp.unlink(missing_ok=True)
                last_err = "invalid_image"
                time.sleep(0.2)
                continue

            os.replace(tmp, final)
            return rk, True, "ok"

        except Exception as e:
            last_err = str(e)
            time.sleep(0.2)

    return rk, False, last_err


def main():
    cfg = load_config("./torob_rag/configs/config.yaml")

    import yaml
    raw = yaml.safe_load(Path("./torob_rag/configs/config.yaml").read_text(encoding="utf-8"))
    run_name = raw.get("runs", {}).get("image_run", "clip_v1")
    dl = raw.get("image_download", {})
    timeout = int(dl.get("timeout_sec", 20))
    max_retries = int(dl.get("max_retries", 2))
    num_workers = int(dl.get("num_workers", 8))
    ua = str(dl.get("user_agent", "torob-rag/0.1"))

    run_dir = cfg.artifacts_dir / "images" / run_name
    subset_path = run_dir / "image_subset.parquet"
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing subset: {subset_path}")

    out_dir = run_dir / "raw_images"
    ensure_dir(out_dir)

    subset = pd.read_parquet(subset_path)
    items = list(zip(subset["random_key"].astype(str).tolist(), subset["image_url"].astype(str).tolist()))

    ok = 0
    fail = 0
    errors = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(download_one, session, rk, url, out_dir, timeout, max_retries, ua) for rk, url in items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                rk, success, msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    errors.append({"random_key": rk, "error": msg})

    summary: Dict[str, Any] = {
        "run": run_name,
        "attempted": int(len(items)),
        "download_ok": int(ok),
        "download_fail": int(fail),
        "raw_images_dir": str(out_dir),
    }
    (run_dir / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors:
        pd.DataFrame(errors).to_csv(run_dir / "download_failures.csv", index=False)

    rprint("[bold green]Checkpoint 08B complete[/bold green]")
    rprint(summary)


if __name__ == "__main__":
    main()
