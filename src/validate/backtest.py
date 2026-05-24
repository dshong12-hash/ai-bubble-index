"""Crash-event backtest: how did the index signal before known market tops?

Three reference events:
  dot_com       — NASDAQ/SPX top 2000-03-24, bottom 2002-10-09  (~49 % SPX drawdown)
  gfc           — SPX top 2007-10-09, bottom 2009-03-09         (~57 % SPX drawdown)
  post_covid    — SPX ATH 2022-01-03, bottom 2022-10-12         (~25 % SPX drawdown)
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from src.common import get_logger, load_raw
from src.transforms.score import regime_label

log = get_logger(__name__)

CRASH_EVENTS: list[tuple[str, str, str, str]] = [
    # (label, pre-crash window start, crash top, crash bottom)
    ("dot_com",    "1998-01-01", "2000-03-24", "2002-10-09"),
    ("gfc",        "2005-01-01", "2007-10-09", "2009-03-09"),
    ("post_covid", "2020-06-01", "2022-01-03", "2022-10-12"),
]


def _spx_drawdown(top: pd.Timestamp, bottom: pd.Timestamp) -> float | None:
    """Compute SPX drawdown from top to bottom using collected Yahoo data."""
    try:
        yahoo = load_raw("yahoo")
        spx = yahoo["^GSPC"].dropna()
        top_val   = spx.asof(top)
        bot_val   = spx.asof(bottom)
        if pd.isna(top_val) or pd.isna(bot_val) or top_val == 0:
            return None
        return round((bot_val - top_val) / top_val * 100, 1)
    except Exception:
        return None


def analyze(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each crash event report:
      - index level at crash top
      - peak index in pre-crash window (+ date, regime, lead days)
      - index level at crash bottom
      - SPX drawdown for reference
    """
    composite = scores_df["bubble_index"].dropna().sort_index()
    rows = []

    for label, pre_start, crash_top_s, crash_bottom_s in CRASH_EVENTS:
        pre   = pd.Timestamp(pre_start)
        top   = pd.Timestamp(crash_top_s)
        bot   = pd.Timestamp(crash_bottom_s)

        window = composite[(composite.index >= pre) & (composite.index <= top)]
        if window.empty:
            log.warning(f"[backtest] No composite data for '{label}' window — skipping")
            continue

        peak_date = window.idxmax()
        peak_val  = window.max()
        at_top    = composite.asof(top)
        at_bot    = composite.asof(bot)
        lead_days = (top - peak_date).days

        rows.append({
            "event":              label,
            "crash_top":          top.date(),
            "crash_bottom":       bot.date(),
            "index_peak_date":    peak_date.date(),
            "index_peak":         round(peak_val, 1),
            "regime_at_peak":     regime_label(peak_val),
            "lead_days":          lead_days,
            "index_at_crash_top": round(at_top, 1) if not pd.isna(at_top) else None,
            "index_at_bottom":    round(at_bot, 1)  if not pd.isna(at_bot)  else None,
            "spx_drawdown_pct":   _spx_drawdown(top, bot),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info(f"[backtest] Analyzed {len(df)} crash events:\n{df.to_string(index=False)}")
    return df
