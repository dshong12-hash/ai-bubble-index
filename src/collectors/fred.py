"""FRED data collector.

Uses the public FRED CSV download endpoint (no API key required for the
fredgraph.csv path used here). For higher rate limits or for batch use,
set a FRED API key via env var FRED_API_KEY and install `fredapi`.

Series collected:
  - DGS30 : US 30Y Treasury
  - DGS10 : US 10Y Treasury
  - DFII10: US 10Y Real Yield (TIPS)
  - BAMLH0A0HYM2: ICE BofA US HY OAS
  - BAMLH0A3HYC : ICE BofA CCC HY OAS
"""
from __future__ import annotations
from io import StringIO
import os
import time
import requests
import pandas as pd

from src.common import get_logger, save_raw

log = get_logger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "DGS30":        "US 30Y Treasury",
    "DGS10":        "US 10Y Treasury",
    "DFII10":       "US 10Y Real Yield (TIPS)",
    "BAMLH0A0HYM2": "HY OAS (ICE BofA)",
    "BAMLH0A3HYC":  "CCC HY OAS",
}


def _fetch_via_csv(code: str, retries: int = 3, sleep: float = 1.5) -> pd.Series:
    url = FRED_CSV_URL.format(code=code)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "ai-bubble-index/1.0"})
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            # FRED CSVs may use lowercase 'observation_date' (2024+) or the series code.
            date_col = next((c for c in df.columns if c.lower() in ("observation_date", "date")), df.columns[0])
            val_col = [c for c in df.columns if c != date_col][0]
            df[date_col] = pd.to_datetime(df[date_col])
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            s = df.set_index(date_col)[val_col].rename(code).sort_index().dropna()
            return s
        except Exception as e:
            last_exc = e
            log.warning(f"FRED CSV fetch failed for {code} (attempt {attempt}/{retries}): {e}")
            time.sleep(sleep * attempt)
    raise RuntimeError(f"FRED fetch failed for {code}") from last_exc


def _fetch_via_api(code: str, api_key: str) -> pd.Series:
    params = {
        "series_id": code,
        "api_key": api_key,
        "file_type": "json",
    }
    r = requests.get(FRED_API_URL, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        raise RuntimeError(f"No observations for {code}")
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].rename(code).sort_index().dropna()


def fetch_series(code: str) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        try:
            return _fetch_via_api(code, api_key)
        except Exception as e:
            log.warning(f"FRED API fetch failed for {code}, falling back to CSV: {e}")
    return _fetch_via_csv(code)


def collect_all() -> pd.DataFrame:
    """Fetch all configured FRED series and return as a wide dataframe."""
    cols = {}
    for code, label in FRED_SERIES.items():
        log.info(f"FRED fetching {code} ({label})")
        try:
            cols[code] = fetch_series(code)
        except Exception as e:
            log.error(f"FRED fetch FAILED for {code}: {e}")
    if not cols:
        raise RuntimeError("All FRED fetches failed.")
    df = pd.concat(cols.values(), axis=1)
    df.index.name = "date"
    return df


def main():
    df = collect_all()
    path = save_raw(df, "fred")
    log.info(f"Saved {len(df)} rows × {df.shape[1]} series → {path}")
    log.info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    log.info(f"Latest values:\n{df.dropna(how='all').tail(1).T}")


if __name__ == "__main__":
    main()
