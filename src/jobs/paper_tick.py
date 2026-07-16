"""Job: paper_tick — simulated trading tick (rules-based).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Usage:
  python -m src.jobs.paper_tick
  python -m src.jobs.paper_tick --date 2026-07-08
  python -m src.jobs.paper_tick --date 2026-07-08 --force-price 341.0
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from src.paper.tick import run_paper_tick

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper trading tick")
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument("--force-price", type=float, help="Override quote for demo/tests")
    parser.add_argument(
        "--phase",
        choices=("premarket", "open", "closed"),
        help="Override session phase",
    )
    parser.add_argument(
        "--allow-non-trading-day",
        action="store_true",
        help="Allow weekends/holidays (demo)",
    )
    args = parser.parse_args()
    d = date.fromisoformat(args.date) if args.date else None
    result = run_paper_tick(
        d,
        session_phase=args.phase,
        force_price=args.force_price,
        allow_non_trading_day=args.allow_non_trading_day,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
