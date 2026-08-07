"""Phase 3 entry point — backtest + sensitivity analysis.

Usage:
    python -m src.run_validate
"""
from __future__ import annotations
import sys, traceback
import pandas as pd

from src.common import get_logger, DATA_PROCESSED
from src.validate.backtest import analyze as backtest, threshold_drawdowns
from src.validate.sensitivity import analyze as sensitivity

log = get_logger(__name__)


def main():
    log.info("Starting Phase 3: Validate")
    try:
        scores_df = pd.read_parquet(DATA_PROCESSED / "scores.parquet")
        pillar_df = scores_df.drop(columns=["bubble_index"], errors="ignore")

        log.info("=== Backtest ===")
        bt = backtest(scores_df)
        bt.to_csv(DATA_PROCESSED / "backtest.csv", index=False)
        log.info(f"Saved → {DATA_PROCESSED / 'backtest.csv'}")

        log.info("=== Threshold (≥70) drawdowns ===")
        td = threshold_drawdowns(scores_df)
        td.to_csv(DATA_PROCESSED / "threshold_drawdowns.csv", index=False)
        log.info(f"Saved → {DATA_PROCESSED / 'threshold_drawdowns.csv'}")

        log.info("=== Sensitivity ===")
        sv = sensitivity(pillar_df)
        sv.to_csv(DATA_PROCESSED / "sensitivity.csv", index=False)
        log.info(f"Saved → {DATA_PROCESSED / 'sensitivity.csv'}")

        log.info("Phase 3 complete.")
    except Exception as e:
        log.error(f"Phase 3 failed: {e}")
        log.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
