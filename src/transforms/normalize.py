"""Rolling percentile normalization + direction flip → 0-100 per metric.

Each metric is ranked within its own 24-month trailing window so the score
reflects *current conditions relative to recent history*, not absolute levels.
Direction: +1 → high value = bubble signal; -1 → low value = bubble signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.common import get_logger

log = get_logger(__name__)

METRIC_DIRECTION: dict[str, int] = {
    "spx_vs_rsp_12m":   +1,
    "us_30y_yield":     +1,
    "us_10y_yield":     +1,
    "us_10y_real":      +1,
    "term_premium_10y": +1,
    "move_index":       +1,
    "hy_oas":           -1,  # tight spread = froth
    "ccc_spread":       -1,
    "bdc_etf":          +1,
    "leveraged_loan":   +1,
    "ipo_etf":          +1,
    "ipo_etf_relative": +1,
    "ritter_first_day": +1,
}

METRIC_PILLAR: dict[str, str] = {
    "spx_vs_rsp_12m":   "concentration",
    "us_30y_yield":     "bond_vigilantes",
    "us_10y_yield":     "bond_vigilantes",
    "us_10y_real":      "bond_vigilantes",
    "term_premium_10y": "bond_vigilantes",
    "move_index":       "bond_vigilantes",
    "hy_oas":           "private_credit",
    "ccc_spread":       "private_credit",
    "bdc_etf":          "private_credit",
    "leveraged_loan":   "private_credit",
    "ipo_etf":          "ipo_saturation",
    "ipo_etf_relative": "ipo_saturation",
    "ritter_first_day": "ipo_saturation",
}


def _rolling_pct_rank(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Percentile rank of the last value in the rolling window (0–100)."""
    def _rank(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.nan
        last = valid[-1]
        below = np.sum(valid[:-1] < last)
        equal = np.sum(valid[:-1] == last)
        return float(below + 0.5 * equal) / (len(valid) - 1) * 100.0

    return s.rolling(window=window, min_periods=min_periods).apply(_rank, raw=True)


def normalize(metrics_df: pd.DataFrame, window_days: int = 504) -> pd.DataFrame:
    """
    Rolling percentile rank each metric then flip direction so 100 = max bubble signal.
    window_days ≈ 24 months of trading days (default 504).
    """
    min_periods = max(window_days // 4, 30)
    out = pd.DataFrame(index=metrics_df.index)

    for col in metrics_df.columns:
        direction = METRIC_DIRECTION.get(col, +1)
        pct = _rolling_pct_rank(
            metrics_df[col].astype(float),
            window=window_days,
            min_periods=min_periods,
        )
        out[col] = 100.0 - pct if direction == -1 else pct

    log.info(f"Normalized {len(out.columns)} metrics (window={window_days}d, min={min_periods}d)")
    return out
