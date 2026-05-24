"""Phase 2 entry point — transform & score.

Usage:
    python -m src.run_score
"""
from __future__ import annotations
import sys
import traceback

from src.common import get_logger
from src.transforms.pipeline import run

log = get_logger(__name__)


def main():
    log.info("Starting Phase 2: Transform & Score")
    try:
        run()
        log.info("Phase 2 complete.")
    except Exception as e:
        log.error(f"Phase 2 failed: {e}")
        log.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
