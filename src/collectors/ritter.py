"""Jay Ritter IPO statistics collector.

Prof. Jay Ritter (UFL) publishes a widely-cited monthly IPO dataset, updated
periodically. The exact file layout has evolved; we look for common columns:
  - Year, Month (or Date)
  - Number of IPOs
  - Avg/Median 1st day return
  - Avg/Median Money Left on Table

Because the URL/format has changed historically, we degrade gracefully:
  1) Try the canonical UFL pages
  2) Fall back to a manual file at data/raw/ritter_ipo.xlsx
  3) If neither available, return an empty df with a warning (Pillar 4 still
     works via Yahoo IPO/SPY relative strength)
"""
from __future__ import annotations
import io
import requests
import pandas as pd

from src.common import get_logger, save_raw, DATA_RAW

log = get_logger(__name__)

CANDIDATE_URLS = [
    "https://site.warrington.ufl.edu/ritter/files/IPOs-Monthly.pdf",  # PDF table fallback (not parsed)
    "https://site.warrington.ufl.edu/ritter/files/IPO-Statistics.pdf",
]


def fetch_manual() -> pd.DataFrame | None:
    for ext in ("xlsx", "xls", "csv"):
        path = DATA_RAW / f"ritter_ipo.{ext}"
        if path.exists():
            log.info(f"Loading Ritter manual file: {path}")
            if ext == "csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            return _normalize(df)
    return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and produce a monthly DatetimeIndex."""
    cols_lower = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # Date construction
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "year" in df.columns and "month" in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    elif "year" in df.columns:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31", errors="coerce")
    else:
        raise ValueError(f"Could not infer date from Ritter columns: {list(df.columns)}")

    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # Keep numeric columns of interest
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols]
    df.index.name = "date"
    return df


def collect_all() -> pd.DataFrame:
    manual = fetch_manual()
    if manual is not None and not manual.empty:
        return manual

    log.warning(
        "Ritter IPO data not available automatically. "
        "Download from https://site.warrington.ufl.edu/ritter/ipo-data/ "
        "and place at data/raw/ritter_ipo.xlsx. "
        "Continuing with Pillar 4 using only Yahoo IPO ETF data."
    )
    # Return an empty DataFrame with date index so downstream code does not crash.
    return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))


def main():
    df = collect_all()
    if df.empty:
        log.warning("Ritter dataset empty — no file saved.")
        return
    path = save_raw(df, "ritter_ipo")
    log.info(f"Saved {len(df)} rows × {df.shape[1]} cols → {path}")
    log.info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")


if __name__ == "__main__":
    main()
