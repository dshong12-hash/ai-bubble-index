"""Aggregate normalized metrics → pillar scores → composite bubble index."""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml

from src.common import get_logger, CONFIG_DIR
from src.transforms.normalize import METRIC_PILLAR

log = get_logger(__name__)


def _cfg() -> dict:
    with open(CONFIG_DIR / "weights.yaml") as f:
        return yaml.safe_load(f)


def pillar_scores(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight mean of available metrics within each pillar → 0-100."""
    pillars: dict[str, pd.Series] = {}
    for pillar in sorted(set(METRIC_PILLAR.values())):
        cols = [c for c in normalized_df.columns if METRIC_PILLAR.get(c) == pillar]
        if cols:
            pillars[pillar] = normalized_df[cols].mean(axis=1, skipna=True)
        else:
            log.warning(f"No metrics available for pillar '{pillar}'")

    if not pillars:
        raise RuntimeError("No pillar scores could be computed.")

    df = pd.DataFrame(pillars)
    df.index.name = "date"
    return df


def composite_score(pillar_df: pd.DataFrame, ema_span: int | None = None) -> pd.Series:
    """
    Weighted average of pillar scores (re-normalises weights for missing pillars).
    Applies EMA smoothing configured in weights.yaml.
    """
    cfg = _cfg()
    weights: dict[str, float] = cfg["pillar_weights"]
    if ema_span is None:
        ema_span = cfg.get("ema_span_days", 28)

    w_series = pd.Series(weights)
    pillars_present = [p for p in weights if p in pillar_df.columns]
    if not pillars_present:
        raise RuntimeError("No pillar scores match configured weights.")

    sub = pillar_df[pillars_present]
    w_sub = w_series[pillars_present]

    # Row-wise weighted mean; missing pillars in a row don't contribute weight
    valid_w = sub.notna().multiply(w_sub)          # weight where valid, 0 where NaN
    total_w = valid_w.sum(axis=1).replace(0, np.nan)
    composite = sub.fillna(0).multiply(valid_w).sum(axis=1) / total_w
    composite = composite.ewm(span=ema_span, adjust=False).mean()
    composite.name = "bubble_index"

    log.info(f"Composite score latest={composite.dropna().iloc[-1]:.1f} (EMA span={ema_span}d)")
    return composite


def regime_label(score: float) -> str:
    cfg = _cfg().get("regime_thresholds", {})
    if score <= cfg.get("green_max", 30):
        return "green"
    if score <= cfg.get("yellow_max", 55):
        return "yellow"
    if score <= cfg.get("orange_max", 75):
        return "orange"
    return "red"
