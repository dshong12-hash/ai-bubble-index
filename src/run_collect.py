"""Phase 1 orchestrator — run all data collectors.

Usage:
  python -m src.run_collect            # run all
  python -m src.run_collect fred yahoo # run subset
"""
from __future__ import annotations
import sys
import traceback
from typing import Callable

from src.common import get_logger
from src.collectors import fred, yahoo, nyfed, ritter

log = get_logger(__name__)

COLLECTORS: dict[str, Callable] = {
    "fred":   fred.main,
    "yahoo":  yahoo.main,
    "nyfed":  nyfed.main,
    "ritter": ritter.main,
}


def main(names: list[str] | None = None):
    names = names or list(COLLECTORS.keys())
    results: dict[str, str] = {}
    for name in names:
        if name not in COLLECTORS:
            log.warning(f"Unknown collector: {name}")
            continue
        log.info(f"=== Running collector: {name} ===")
        try:
            COLLECTORS[name]()
            results[name] = "OK"
        except Exception as e:
            log.error(f"Collector {name} FAILED: {e}")
            log.debug(traceback.format_exc())
            results[name] = f"FAIL: {e}"

    log.info("=== Collection summary ===")
    for name, status in results.items():
        log.info(f"  {name:10s} → {status}")

    # Exit non-zero if all failed
    if all(s.startswith("FAIL") for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:] or None
    main(args)
