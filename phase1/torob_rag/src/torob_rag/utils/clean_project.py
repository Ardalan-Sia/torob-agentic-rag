from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


# Directories to remove (relative patterns)
COMMON_DIRS = [
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/.ipynb_checkpoints",
    "**/build",
    "**/dist",
    "**/*.egg-info",
]

# Files to remove (relative patterns)
COMMON_FILES = [
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/*.log",
    "**/*.tmp",
    "**/*.temp",
    "**/.DS_Store",
]

# Artifacts subfolders that are safe to remove (lightweight)
ARTIFACTS_LIGHT = [
    "torob_rag/artifacts/eval/**/failure_examples*.csv",
    "torob_rag/artifacts/eval/**/failure_examples_readable.csv",
]

# Dangerous (heavy) artifacts: keep by default
ARTIFACTS_HEAVY_ROOTS = [
    "torob_rag/artifacts/clean",
    "torob_rag/artifacts/embeddings",
    "torob_rag/artifacts/vector_db",
    "torob_rag/artifacts/subset",
    "torob_rag/artifacts/eda",
    "torob_rag/artifacts/checkpoint_*.json",
]


def iter_matches(root: Path, pattern: str) -> Iterable[Path]:
    # glob supports ** patterns
    for p in root.glob(pattern):
        yield p


def remove_path(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def is_inside_any(path: Path, roots: List[Path]) -> bool:
    try:
        path_res = path.resolve()
    except Exception:
        path_res = path
    for r in roots:
        try:
            r_res = r.resolve()
        except Exception:
            r_res = r
        if r_res in path_res.parents or path_res == r_res:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Clean project caches/tmp files safely.")
    ap.add_argument("--mode", choices=["safe", "light", "aggressive"], default="safe",
                    help="safe: caches only | light: + remove small eval csvs | aggressive: delete artifacts/ (rebuild required)")
    ap.add_argument("--yes", action="store_true", help="Actually delete files. Without this, it will just print a dry-run.")
    ap.add_argument("--project-root", default=".", help="Project root (default: current dir)")
    ap.add_argument("--remove-venv", action="store_true", help="Also remove .venv directory (NOT recommended)")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    # Protect heavy artifacts by default unless aggressive
    heavy_roots = [root / p for p in ARTIFACTS_HEAVY_ROOTS]
    venv_path = root / ".venv"

    targets: List[Path] = []

    # Common dirs/files
    for pat in COMMON_DIRS:
        targets.extend(list(iter_matches(root, pat)))
    for pat in COMMON_FILES:
        targets.extend(list(iter_matches(root, pat)))

    # Light mode removes some small eval csv outputs (optional)
    if args.mode in ("light", "aggressive"):
        for pat in ARTIFACTS_LIGHT:
            targets.extend(list(iter_matches(root, pat)))

    # Aggressive: delete whole artifacts folder (everything will be regenerated)
    if args.mode == "aggressive":
        art = root / "torob_rag" / "artifacts"
        if art.exists():
            targets.append(art)

    # Optionally remove venv
    if args.remove_venv and venv_path.exists():
        targets.append(venv_path)

    # De-dup + sort stable
    uniq = sorted({p for p in targets if p.exists()}, key=lambda x: str(x))

    # In safe/light modes, do NOT delete heavy artifacts subtrees
    if args.mode in ("safe", "light"):
        filtered = []
        for p in uniq:
            # never touch .git
            if ".git" in p.parts:
                continue
            if is_inside_any(p, heavy_roots):
                continue
            filtered.append(p)
        uniq = filtered

    if not uniq:
        print("Nothing to clean.")
        return

    print(f"Mode={args.mode} | Dry-run={not args.yes}")
    print("Will remove:")
    for p in uniq:
        print(" -", p)

    if not args.yes:
        print("\nDry-run only. Re-run with --yes to actually delete.")
        return

    for p in uniq:
        remove_path(p)

    print("\n✅ Clean complete.")


if __name__ == "__main__":
    main()
