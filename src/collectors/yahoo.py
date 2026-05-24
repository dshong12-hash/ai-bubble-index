"""Yahoo Finance data collector via yfinance.

Tickers collected:
  - ^GSPC : S&P 500
  - RSP   : Invesco S&P 500 Equal Weight ETF
  - ^MOVE : MOVE Index (bond volatility)
  - BIZD  : VanEck BDC Income ETF (private credit proxy)
  - BKLN  : Invesco Senior Loan ETF (leveraged loan proxy)
  - IPO   : Renaissance IPO ETF
  - SPY   : SPDR S&P 500 (benchmark for IPO relative strength)
"""
from __future__ import annotations
import time
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("yfinance not installed. Run: pip install yfinance --break-system-packages") from e

from src.common import get_logger, save_raw

log = get_logger(__name__)

YAHOO_TICKERS = {
    "^GSPC": "S&P 500",
    "RSP":   "S&P 500 Equal Weight",
    "^MOVE": "MOVE Index",
    "BIZD":  "BDC Income ETF",
    "BKLN":  "Senior Loan ETF",
    "IPO":   "Renaissance IPO ETF",
    "SPY":   "SPDR S&P 500",
}

START_DATE = "2000-01-01"


def fetch_one(ticker: str, start: str = START_DATE, retries: int = 3) -> pd.Series:
    """Fetch adjusted close prices for a single ticker."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, progress=False,
                auto_adjust=True, threads=False
            )
            if df is None or df.empty:
                raise RuntimeError(f"empty dataframe for {ticker}")
            # yfinance >=0.2 returns MultiIndex columns when multiple tickers requested;
            # we requested single ticker but defend anyway.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            col = "Close" if "Close" in df.columns else df.columns[0]
            s = df[col].rename(ticker).sort_index().dropna()
            if s.empty:
                raise RuntimeError(f"all NaN for {ticker}")
            return s
        except Exception as e:
            last_exc = e
            log.warning(f"Yahoo fetch failed for {ticker} (attempt {attempt}/{retries}): {e}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Yahoo fetch failed for {ticker}") from last_exc


def collect_all() -> pd.DataFrame:
    cols = {}
    for ticker, label in YAHOO_TICKERS.items():
        log.info(f"Yahoo fetching {ticker} ({label})")
        try:
            cols[ticker] = fetch_one(ticker)
        except Exception as e:
            log.error(f"Yahoo fetch FAILED for {ticker}: {e}")
    if not cols:
        raise RuntimeError("All Yahoo fetches failed.")
    df = pd.concat(cols.values(), axis=1)
    df.index.name = "date"
    # Yahoo returns tz-aware dates in some versions; normalize.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def main():
    df = collect_all()
    path = save_raw(df, "yahoo")
    log.info(f"Saved {len(df)} rows × {df.shape[1]} tickers → {path}")
    log.info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    log.info(f"Latest values:\n{df.dropna(how='all').tail(1).T}")


if __name__ == "__main__":
    main()
