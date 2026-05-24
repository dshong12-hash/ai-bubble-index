"""Phase 2 orchestrator: raw data → metrics → normalised → pillar scores → composite."""
from __future__ import annotations
import yaml
import pandas as pd

from src.common import get_logger, DATA_PROCESSED, DATA_SNAPSHOTS, CONFIG_DIR
from src.transforms.metrics import compute as compute_metrics
from src.transforms.normalize import normalize
from src.transforms.score import pillar_scores, composite_score, regime_label

log = get_logger(__name__)

PILLAR_LABELS = {
    "concentration":  "주도주 압착",
    "bond_vigilantes": "채권 자경단",
    "private_credit": "사모 크레딧",
    "ipo_saturation": "IPO 포화",
}


def run(start: str = "2002-01-01") -> pd.DataFrame:
    cfg_path = CONFIG_DIR / "weights.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    window_days = cfg.get("rolling_window_months", 24) * 21
    ema_span = cfg.get("ema_span_days", 28)

    # ── Step 1: metrics ──────────────────────────────────────────────────────
    log.info("Step 1/3 — computing raw metrics")
    metrics_df = compute_metrics(start=start)
    metrics_df.to_parquet(DATA_PROCESSED / "metrics.parquet")

    # ── Step 2: normalise ────────────────────────────────────────────────────
    log.info(f"Step 2/3 — rolling percentile normalisation (window={window_days}d)")
    norm_df = normalize(metrics_df, window_days=window_days)
    norm_df.to_parquet(DATA_PROCESSED / "normalized.parquet")

    # ── Step 3: score ────────────────────────────────────────────────────────
    log.info("Step 3/3 — pillar aggregation + composite score")
    pillar_df = pillar_scores(norm_df)
    composite = composite_score(pillar_df, ema_span=ema_span)

    scores_df = pillar_df.copy()
    scores_df["bubble_index"] = composite
    scores_df.to_parquet(DATA_PROCESSED / "scores.parquet")

    # ── Snapshot ─────────────────────────────────────────────────────────────
    valid = scores_df.dropna(subset=["bubble_index"])
    if valid.empty:
        log.warning("No valid composite scores — snapshot not saved.")
        return scores_df

    latest = valid.iloc[-1]
    snap = valid.tail(1).copy()
    snap["regime"] = regime_label(latest["bubble_index"])
    snap_path = DATA_SNAPSHOTS / f"{latest.name.date()}.csv"
    snap.to_csv(snap_path)

    # ── Report ────────────────────────────────────────────────────────────────
    index_val = latest["bubble_index"]
    regime = regime_label(index_val)
    regime_emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(regime, "⚪")

    log.info("=" * 52)
    log.info(f"  AI Bubble Index — {latest.name.date()}")
    log.info(f"  종합 지수   : {index_val:5.1f} / 100  {regime_emoji} {regime.upper()}")
    log.info("-" * 52)
    for pillar, label in PILLAR_LABELS.items():
        if pillar in latest and not pd.isna(latest[pillar]):
            log.info(f"  {label:12s}: {latest[pillar]:5.1f}")
    log.info("=" * 52)
    log.info(f"  Snapshot → {snap_path}")

    return scores_df
