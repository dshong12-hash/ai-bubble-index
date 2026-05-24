"""Common utilities for data collectors."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SNAPSHOTS = PROJECT_ROOT / "data" / "snapshots"
CONFIG_DIR = PROJECT_ROOT / "config"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_SNAPSHOTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---- IO helpers ----
def save_raw(df: pd.DataFrame, name: str) -> Path:
    """Save a raw collected dataframe to parquet under data/raw/."""
    if df is None or df.empty:
        raise ValueError(f"Cannot save empty dataframe: {name}")
    df = df.copy()
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    path = DATA_RAW / f"{name}.parquet"
    df.to_parquet(path)
    return path


def load_raw(name: str) -> pd.DataFrame:
    path = DATA_RAW / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
