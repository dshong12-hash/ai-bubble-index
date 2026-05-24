"""NY Fed ACM Term Premium data collector.

The Adrian-Crump-Moench (ACM) term premium estimates are published as a
freely downloadable Excel/CSV file from the NY Fed. We try a few known
locations; if all fail, the user should download manually and place at
data/raw/acm_termpremium.xlsx.

Series of interest:
  - ACMTP10 : 10-Year zero-coupon term premium
  - ACMY10  : 10-Year fitted yield (for cross-check)
"""
from __future__ import annotations
import io
import requests
import pandas as pd

from src.common import get_logger, save_raw, DATA_RAW

log = get_logger(__name__)

# Known candidate URLs (NY Fed has moved this file over time).
CANDIDATE_URLS = [
    "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls",
    "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xlsx",
    "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.csv",
]

TARGET_COLS = ["ACMTP10", "ACMY10", "ACMRNY10"]  # take whichever are present


def _read_any(content: bytes, fname: str) -> pd.DataFrame:
    """Read xls/xlsx/csv into a dataframe with a date index."""
    bio = io.BytesIO(content)
    lower = fname.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(bio)
    elif lower.endswith(".xls"):
        # Older NY Fed file is xls; needs xlrd. Try openpyxl as fallback.
        try:
            df = pd.read_excel(bio, engine="xlrd")
        except Exception:
            bio.seek(0)
            df = pd.read_excel(bio, engine="openpyxl")
    else:
        df = pd.read_excel(bio, engine="openpyxl")

    # Find the date column.
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("date", "asof", "observation_date"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    return df


def fetch_acm() -> pd.DataFrame:
    last_exc = None
    for url in CANDIDATE_URLS:
        try:
            log.info(f"NY Fed trying {url}")
            r = requests.get(url, timeout=60, headers={"User-Agent": "ai-bubble-index/1.0"})
            r.raise_for_status()
            fname = url.split("/")[-1]
            df = _read_any(r.content, fname)
            log.info(f"NY Fed ACM loaded: {df.shape}, cols={list(df.columns)[:6]}...")
            return df
        except Exception as e:
            last_exc = e
            log.warning(f"NY Fed URL failed: {url}: {e}")

    # Last resort — check for manual file on disk
    manual = DATA_RAW / "acm_termpremium.xlsx"
    if manual.exists():
        log.info(f"Loading manual file: {manual}")
        with open(manual, "rb") as f:
            return _read_any(f.read(), manual.name)

    raise RuntimeError(
        "NY Fed ACM term premium not available from any known URL. "
        "Download manually from newyorkfed.org and place at data/raw/acm_termpremium.xlsx"
    ) from last_exc


def collect_all() -> pd.DataFrame:
    df = fetch_acm()
    keep = [c for c in df.columns if c in TARGET_COLS]
    if not keep:
        # Some versions use slightly different headers; keep all numeric columns.
        keep = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        log.warning(f"Target columns not found; kept all numeric cols: {keep}")
    return df[keep]


def main():
    df = collect_all()
    path = save_raw(df, "nyfed_acm")
    log.info(f"Saved {len(df)} rows × {df.shape[1]} series → {path}")
    log.info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    log.info(f"Latest:\n{df.dropna(how='all').tail(1).T}")


if __name__ == "__main__":
    main()
