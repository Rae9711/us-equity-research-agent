from __future__ import annotations

import argparse
import logging

from src.collectors.step0 import run_step0
from src.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 0 raw data collection")
    parser.add_argument("--date", help="Trading date YYYY-MM-DD (default: today ET)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PIT snapshot",
    )
    args = parser.parse_args()

    init_db()
    trading_date = None
    if args.date:
        from datetime import date

        trading_date = date.fromisoformat(args.date)

    payload = run_step0(trading_date, force=args.force)
    print(payload["conclusion"]["judgment"])
    print(payload["conclusion"]["one_liner"])


if __name__ == "__main__":
    main()
