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

# Alert threshold — index level that historically precedes major crashes.
ALERT_THRESHOLD = 70.0


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

        # First time the index crossed the alert threshold (≥70) before the top.
        # This is an earlier, more conservative signal than the index peak.
        above_thr   = window[window >= ALERT_THRESHOLD]
        if not above_thr.empty:
            first_cross_date = above_thr.index[0]
            first_cross_lead = (top - first_cross_date).days
            first_cross_idx  = round(float(above_thr.iloc[0]), 1)
        else:
            first_cross_date = None
            first_cross_lead = None
            first_cross_idx  = None

        rows.append({
            "event":                  label,
            "crash_top":              top.date(),
            "crash_bottom":           bot.date(),
            "index_peak_date":        peak_date.date(),
            "index_peak":             round(peak_val, 1),
            "regime_at_peak":         regime_label(peak_val),
            "lead_days":              lead_days,
            "first_cross_date":       first_cross_date.date() if first_cross_date is not None else None,
            "first_cross_lead_days":  first_cross_lead,
            "index_at_first_cross":   first_cross_idx,
            "index_at_crash_top":     round(at_top, 1) if not pd.isna(at_top) else None,
            "index_at_bottom":        round(at_bot, 1)  if not pd.isna(at_bot)  else None,
            "spx_drawdown_pct":       _spx_drawdown(top, bot),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info(f"[backtest] Analyzed {len(df)} crash events:\n{df.to_string(index=False)}")
    return df


def threshold_drawdowns(
    scores_df: pd.DataFrame,
    threshold: float = ALERT_THRESHOLD,
    horizons_months: tuple[int, ...] = (6, 12, 24),
) -> pd.DataFrame:
    """SPX drawdown statistics after the index crosses up through ``threshold``.

    Methodology (fully data-driven, reproducible):
      1. Find every distinct *upward* crossing of the threshold (index moves
         from below to ≥ threshold). Each such date is an "entry".
      2. For each horizon (in months), measure the SPX return from the entry
         level to its lowest close within that calendar window — i.e. the
         worst drawdown an investor would face after acting on the signal.
      3. Aggregate across all entries: average, median and worst-case.

    Returns one row per horizon, ready to persist as CSV for the dashboard.
    """
    bi = scores_df["bubble_index"].dropna().sort_index()
    try:
        yahoo = load_raw("yahoo")
        spx = yahoo["^GSPC"].dropna().sort_index()
    except Exception:
        log.warning("[threshold_dd] Yahoo SPX unavailable — skipping")
        return pd.DataFrame()

    above    = bi >= threshold
    cross_up = above & ~above.shift(1, fill_value=False)
    entries  = bi.index[cross_up]

    rows = []
    for months in horizons_months:
        dds: list[float] = []
        for entry in entries:
            start_val = spx.asof(entry)
            if pd.isna(start_val) or start_val == 0:
                continue
            window_end = entry + pd.Timedelta(days=int(round(months * 30.44)))
            fut = spx[(spx.index >= entry) & (spx.index <= window_end)]
            if fut.empty:
                continue
            dd = (fut.min() - start_val) / start_val * 100
            dds.append(dd)

        rows.append({
            "horizon_months":      months,
            "n_entries":           len(dds),
            "avg_drawdown_pct":    round(float(np.mean(dds)), 1)   if dds else None,
            "median_drawdown_pct": round(float(np.median(dds)), 1) if dds else None,
            "worst_drawdown_pct":  round(float(np.min(dds)), 1)    if dds else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info(
            f"[threshold_dd] {int(above.sum())} days ≥{threshold:.0f}, "
            f"{len(entries)} entries:\n{df.to_string(index=False)}"
        )
    return df
