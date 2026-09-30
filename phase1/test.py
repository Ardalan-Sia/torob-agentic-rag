import pandas as pd
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

base_products = Path(__file__).resolve().parent / "Torob-Database" / "base_products.parquet"
print(pd.read_parquet(base_products, engine="pyarrow").head(2).to_dict())
