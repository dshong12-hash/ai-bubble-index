"""Compute raw metric values from Phase-1 parquet files.

All output series are reindexed to a common business-daily calendar
and forward-filled up to 10 days to bridge weekends / holidays /
monthly-release gaps.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from src.common import get_logger, load_raw, DATA_RAW

log = get_logger(__name__)


def _to_bday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    idx = pd.bdate_range(df.index.min(), df.index.max())
    return df.reindex(idx).ffill(limit=10)


def compute(start: str | None = None) -> pd.DataFrame:
    """Return wide DataFrame of raw metric values aligned to business days."""
    fred = _to_bday(load_raw("fred"))
    yahoo = _to_bday(load_raw("yahoo"))
    nyfed = _to_bday(load_raw("nyfed_acm"))

    m: dict[str, pd.Series] = {}

    # ── Pillar 1: Concentration ──────────────────────────────────────────────
    if {"^GSPC", "RSP"}.issubset(yahoo.columns):
        spx_12m = yahoo["^GSPC"].pct_change(252)
        rsp_12m = yahoo["RSP"].pct_change(252)
        m["spx_vs_rsp_12m"] = (spx_12m - rsp_12m) * 100  # percentage-point spread
    # top10_weight (slickcharts scrape) deferred to Phase 4

    # ── Pillar 2: Bond Vigilantes ────────────────────────────────────────────
    for src_col, metric in [
        ("DGS30",  "us_30y_yield"),
        ("DGS10",  "us_10y_yield"),
        ("DFII10", "us_10y_real"),
    ]:
        if src_col in fred.columns:
            m[metric] = fred[src_col]

    if "ACMTP10" in nyfed.columns:
        m["term_premium_10y"] = nyfed["ACMTP10"]

    if "^MOVE" in yahoo.columns:
        m["move_index"] = yahoo["^MOVE"]

    # ── Pillar 3: Private Credit ─────────────────────────────────────────────
    for src_col, metric in [
        ("BAMLH0A0HYM2", "hy_oas"),
        ("BAMLH0A3HYC",  "ccc_spread"),
    ]:
        if src_col in fred.columns:
            m[metric] = fred[src_col]

    for src_col, metric in [
        ("BIZD", "bdc_etf"),
        ("BKLN", "leveraged_loan"),
    ]:
        if src_col in yahoo.columns:
            m[metric] = yahoo[src_col]

    # ── Pillar 4: IPO Saturation ─────────────────────────────────────────────
    if "IPO" in yahoo.columns:
        m["ipo_etf"] = yahoo["IPO"]

    if {"IPO", "SPY"}.issubset(yahoo.columns):
        m["ipo_etf_relative"] = (yahoo["IPO"] / yahoo["SPY"]) * 100

    ritter_path = DATA_RAW / "ritter_ipo.xlsx"
    if ritter_path.exists():
        try:
            rdf = pd.read_excel(ritter_path, index_col=0, parse_dates=True)
            col = next((c for c in rdf.columns if "first" in c.lower()), rdf.columns[0])
            m["ritter_first_day"] = rdf[col]
        except Exception as e:
            log.warning(f"Could not load Ritter IPO data: {e}")

    if not m:
        raise RuntimeError("No metrics computed — check that raw parquet files exist.")

    out = pd.concat(m.values(), axis=1, keys=m.keys())
    out.index.name = "date"
    out = out.sort_index()

    if start:
        out = out[out.index >= pd.Timestamp(start)]

    log.info(f"Metrics: {list(out.columns)}, rows={len(out)}")
    return out
