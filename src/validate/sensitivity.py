"""Pillar-weight sensitivity analysis.

Shifts each pillar weight by ±DELTA (default 5 pp), redistributing equally
among the other pillars, then recomputes the composite score.  Shows which
pillars drive the current reading most.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml

from src.common import get_logger, CONFIG_DIR

log = get_logger(__name__)

DELTA = 0.05  # ±5 percentage points


def _perturb(base: dict[str, float], pillar: str, delta: float) -> dict[str, float]:
    """Shift `pillar` by `delta`, redistribute residual equally among others."""
    w = dict(base)
    others = [p for p in w if p != pillar]
    new_val = max(0.0, min(1.0, w[pillar] + delta))
    actual_delta = new_val - w[pillar]
    w[pillar] = new_val
    for p in others:
        w[p] = max(0.0, w[p] - actual_delta / len(others))
    total = sum(w.values())
    return {p: v / total for p, v in w.items()}


def _composite_latest(pillar_df: pd.DataFrame, weights: dict[str, float], ema_span: int) -> float:
    """Compute the EMA-smoothed composite score at the latest date."""
    w = pd.Series(weights)
    present = [p for p in weights if p in pillar_df.columns]
    sub = pillar_df[present]
    valid_w = sub.notna().multiply(w[present])
    total_w = valid_w.sum(axis=1).replace(0, np.nan)
    composite = sub.fillna(0).multiply(valid_w).sum(axis=1) / total_w
    return float(composite.ewm(span=ema_span, adjust=False).mean().dropna().iloc[-1])


def analyze(pillar_df: pd.DataFrame) -> pd.DataFrame:
    """Return a sensitivity table (one row per pillar)."""
    with open(CONFIG_DIR / "weights.yaml") as f:
        cfg = yaml.safe_load(f)
    base_weights: dict[str, float] = cfg["pillar_weights"]
    ema_span: int = cfg.get("ema_span_days", 28)

    base_score = _composite_latest(pillar_df, base_weights, ema_span)
    rows = []

    for pillar, base_w in base_weights.items():
        w_up = _perturb(base_weights, pillar, +DELTA)
        w_dn = _perturb(base_weights, pillar, -DELTA)
        s_up = _composite_latest(pillar_df, w_up, ema_span)
        s_dn = _composite_latest(pillar_df, w_dn, ema_span)

        current_pillar = (
            round(float(pillar_df[pillar].dropna().iloc[-1]), 1)
            if pillar in pillar_df.columns else None
        )
        rows.append({
            "pillar":         pillar,
            "weight_pct":     round(base_w * 100, 0),
            "pillar_score":   current_pillar,
            "composite_base": round(base_score, 1),
            f"+{int(DELTA*100)}pp":  round(s_up, 1),
            f"-{int(DELTA*100)}pp":  round(s_dn, 1),
            "impact_range":   round(abs(s_up - s_dn), 2),
        })

    df = pd.DataFrame(rows).sort_values("impact_range", ascending=False)
    log.info(f"[sensitivity] base={base_score:.1f}, delta=±{int(DELTA*100)}pp\n{df.to_string(index=False)}")
    return df
