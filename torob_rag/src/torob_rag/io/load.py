from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    data_dir: Path
    artifacts_dir: Path
    tables: Dict[str, str]


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = Path(raw["paths"]["data_dir"]).expanduser().resolve()
    artifacts_dir = Path(raw["paths"]["artifacts_dir"]).expanduser().resolve()
    tables = raw["tables"]

    return ProjectConfig(
        seed=int(raw.get("seed", 42)),
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        tables=tables,
    )


def read_parquet_table(cfg: ProjectConfig, table_key: str) -> pd.DataFrame:
    if table_key not in cfg.tables:
        raise KeyError(f"Unknown table_key={table_key}. Available: {list(cfg.tables.keys())}")

    path = cfg.data_dir / cfg.tables[table_key]
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path)
    return df


def load_all_tables(cfg: ProjectConfig) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for k in cfg.tables.keys():
        out[k] = read_parquet_table(cfg, k)
    return out
