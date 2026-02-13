from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_all_csvs(folder: str | Path) -> dict[str, pd.DataFrame]:
    folder = Path(folder).expanduser()
    csv_paths = sorted(folder.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found in: {folder}")

    tables: dict[str, pd.DataFrame] = {}
    for p in csv_paths:
        name = p.stem.lower()
        tables[name] = pd.read_csv(p)
    return tables
